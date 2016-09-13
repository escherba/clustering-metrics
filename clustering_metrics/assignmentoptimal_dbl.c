/*

 Copyright (c) 2014, Markus Buehren
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are
 met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in
 the documentation and/or other materials provided with the distribution

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.

*/

#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include "assignmentoptimal.h"

#define cell double
#define cell_max INFINITY

void buildassignmentvector_dbl(long*, bool*, long , long );
void computeassignmentcost_dbl(long*, cell*, cell*, long );
void step2a_dbl(long*, cell*, bool*, bool*, bool*, bool*, bool*, long, long, long);
void step2b_dbl(long*, cell*, bool*, bool*, bool*, bool*, bool*, long, long, long);
void step3_dbl (long*, cell*, bool*, bool*, bool*, bool*, bool*, long, long, long);
void step4_dbl (long*, cell*, bool*, bool*, bool*, bool*, bool*, long, long, long, long, long);
void step5_dbl (long*, cell*, bool*, bool*, bool*, bool*, bool*, long, long, long);


void assignmentoptimal_dbl(long *assignment, cell *cost, cell *distMatrixIn, long nOfRows, long nOfColumns)
{
	cell *distMatrix, *distMatrixTemp, *distMatrixEnd, *columnEnd, value, minValue;
	bool *coveredColumns, *coveredRows, *starMatrix, *newStarMatrix, *primeMatrix;
	long nOfElements, minDim, row, col;

	/* initialization */
	*cost = 0;
	for(row=0; row<nOfRows; row++)
		assignment[row] = -1;

	/* create working copy of distance Matrix */
	nOfElements   = nOfRows * nOfColumns;
	distMatrix    = (cell *)malloc(nOfElements * sizeof(cell));
	distMatrixEnd = distMatrix + nOfElements;
    memcpy(distMatrix, distMatrixIn, nOfElements * sizeof(cell));

	/* memory allocation */
	coveredColumns = (bool *)calloc(nOfColumns,  sizeof(bool));
	coveredRows    = (bool *)calloc(nOfRows,     sizeof(bool));
	starMatrix     = (bool *)calloc(nOfElements, sizeof(bool));
	primeMatrix    = (bool *)calloc(nOfElements, sizeof(bool));
	newStarMatrix  = (bool *)calloc(nOfElements, sizeof(bool)); /* used in step4_dbl */


	/* preliminary steps */
	if(nOfRows <= nOfColumns)
	{
		minDim = nOfRows;

		for(row=0; row<nOfRows; row++)
		{
			/* find the smallest element in the row */
			distMatrixTemp = distMatrix + row;
			minValue = *distMatrixTemp;
			distMatrixTemp += nOfRows;
			while(distMatrixTemp < distMatrixEnd)
			{
				value = *distMatrixTemp;
				if(value < minValue)
					minValue = value;
				distMatrixTemp += nOfRows;
			}

			/* subtract the smallest element from each element of the row */
			distMatrixTemp = distMatrix + row;
			while(distMatrixTemp < distMatrixEnd)
			{
				*distMatrixTemp -= minValue;
				distMatrixTemp += nOfRows;
			}
		}

		/* Steps 1 and 2a */
		for(row=0; row<nOfRows; row++)
			for(col=0; col<nOfColumns; col++)
				if(distMatrix[row + nOfRows*col] == 0)
					if(!coveredColumns[col])
					{
						starMatrix[row + nOfRows*col] = true;
						coveredColumns[col]           = true;
						break;
					}
	}
	else /* if(nOfRows > nOfColumns) */
	{
		minDim = nOfColumns;

		for(col=0; col<nOfColumns; col++)
		{
			/* find the smallest element in the column */
			distMatrixTemp = distMatrix     + nOfRows*col;
			columnEnd      = distMatrixTemp + nOfRows;

			minValue = *distMatrixTemp++;
			while(distMatrixTemp < columnEnd)
			{
				value = *distMatrixTemp++;
				if(value < minValue)
					minValue = value;
			}

			/* subtract the smallest element from each element of the column */
			distMatrixTemp = distMatrix + nOfRows*col;
			while(distMatrixTemp < columnEnd)
				*distMatrixTemp++ -= minValue;
		}

		/* Steps 1 and 2a */
		for(col=0; col<nOfColumns; col++)
			for(row=0; row<nOfRows; row++)
				if(distMatrix[row + nOfRows*col] == 0)
					if(!coveredRows[row])
					{
						starMatrix[row + nOfRows*col] = true;
						coveredColumns[col]           = true;
						coveredRows[row]              = true;
						break;
					}
		for(row=0; row<nOfRows; row++)
			coveredRows[row] = false;

	}

	/* move to step 2b */
	step2b_dbl(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);

	/* compute cost and remove invalid assignments */
	computeassignmentcost_dbl(assignment, cost, distMatrixIn, nOfRows);

	/* free allocated memory */
	free(distMatrix);
	free(coveredColumns);
	free(coveredRows);
	free(starMatrix);
	free(primeMatrix);
	free(newStarMatrix);

	return;
}

/********************************************************/
void buildassignmentvector_dbl(long *assignment, bool *starMatrix, long nOfRows, long nOfColumns)
{
	long row, col;

	for(row=0; row<nOfRows; row++)
		for(col=0; col<nOfColumns; col++)
			if(starMatrix[row + nOfRows*col])
			{
				assignment[row] = col;
				break;
			}
}

/********************************************************/
void computeassignmentcost_dbl(long *assignment, cell *cost, cell *distMatrix, long nOfRows)
{
	long row, col;

	for(row=0; row<nOfRows; row++)
	{
		col = assignment[row];
		if(col >= 0)
		{
			*cost += distMatrix[row + nOfRows*col];
		}
	}
}

/********************************************************/
void step2a_dbl(long *assignment, cell *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, long nOfRows, long nOfColumns, long minDim)
{
	bool *starMatrixTemp, *columnEnd;
	long col;

	/* cover every column containing a starred zero */
	for(col=0; col<nOfColumns; col++)
	{
		starMatrixTemp = starMatrix     + nOfRows*col;
		columnEnd      = starMatrixTemp + nOfRows;
		while(starMatrixTemp < columnEnd){
			if(*starMatrixTemp++)
			{
				coveredColumns[col] = true;
				break;
			}
		}
	}

	/* move to step 3 */
	step2b_dbl(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void step2b_dbl(long *assignment, cell *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, long nOfRows, long nOfColumns, long minDim)
{
	long col, nOfCoveredColumns;

	/* count covered columns */
	nOfCoveredColumns = 0;
	for(col=0; col<nOfColumns; col++)
		if(coveredColumns[col])
			nOfCoveredColumns++;

	if(nOfCoveredColumns == minDim)
	{
		/* algorithm finished */
		buildassignmentvector_dbl(assignment, starMatrix, nOfRows, nOfColumns);
	}
	else
	{
		/* move to step 3 */
		step3_dbl(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
	}

}

/********************************************************/
void step3_dbl(long *assignment, cell *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, long nOfRows, long nOfColumns, long minDim)
{
	bool zerosFound;
	long row, col, starCol;

	zerosFound = true;
	while(zerosFound)
	{
		zerosFound = false;
		for(col=0; col<nOfColumns; col++)
			if(!coveredColumns[col])
				for(row=0; row<nOfRows; row++)
					if((!coveredRows[row]) && (distMatrix[row + nOfRows*col] == 0))
					{
						/* prime zero */
						primeMatrix[row + nOfRows*col] = true;

						/* find starred zero in current row */
						for(starCol=0; starCol<nOfColumns; starCol++)
							if(starMatrix[row + nOfRows*starCol])
								break;

						if(starCol == nOfColumns) /* no starred zero found */
						{
							/* move to step 4 */
							step4_dbl(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim, row, col);
							return;
						}
						else
						{
							coveredRows[row]        = true;
							coveredColumns[starCol] = false;
							zerosFound              = true;
							break;
						}
					}
	}

	/* move to step 5 */
	step5_dbl(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void step4_dbl(long *assignment, cell *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, long nOfRows, long nOfColumns, long minDim, long row, long col)
{
	long n, starRow, starCol, primeRow, primeCol;
	long nOfElements = nOfRows*nOfColumns;

	/* create a temporary copy of starMatrix */
    memcpy(newStarMatrix, starMatrix, nOfElements * sizeof(bool));

	/* star current zero */
	newStarMatrix[row + nOfRows*col] = true;

	/* find starred zero in current column */
	starCol = col;
	for(starRow=0; starRow<nOfRows; starRow++)
		if(starMatrix[starRow + nOfRows*starCol])
			break;

	while(starRow<nOfRows)
	{
		/* unstar the starred zero */
		newStarMatrix[starRow + nOfRows*starCol] = false;

		/* find primed zero in current row */
		primeRow = starRow;
		for(primeCol=0; primeCol<nOfColumns; primeCol++)
			if(primeMatrix[primeRow + nOfRows*primeCol])
				break;

		/* star the primed zero */
		newStarMatrix[primeRow + nOfRows*primeCol] = true;

		/* find starred zero in current column */
		starCol = primeCol;
		for(starRow=0; starRow<nOfRows; starRow++)
			if(starMatrix[starRow + nOfRows*starCol])
				break;
	}

	/* use temporary copy as new starMatrix */
	/* delete all primes, uncover all rows */
	for(n=0; n<nOfElements; n++)
	{
		primeMatrix[n] = false;
		starMatrix[n]  = newStarMatrix[n];
	}
	for(n=0; n<nOfRows; n++)
		coveredRows[n] = false;

	/* move to step 2a */
	step2a_dbl(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}

/********************************************************/
void step5_dbl(long *assignment, cell *distMatrix, bool *starMatrix, bool *newStarMatrix, bool *primeMatrix, bool *coveredColumns, bool *coveredRows, long nOfRows, long nOfColumns, long minDim)
{
	cell h, value;
	long row, col;

	/* find smallest uncovered element h */
	h = cell_max;
	for(row=0; row<nOfRows; row++)
		if(!coveredRows[row])
			for(col=0; col<nOfColumns; col++)
				if(!coveredColumns[col])
				{
					value = distMatrix[row + nOfRows*col];
					if(value < h)
						h = value;
				}

	/* add h to each covered row */
	for(row=0; row<nOfRows; row++)
		if(coveredRows[row])
			for(col=0; col<nOfColumns; col++)
				distMatrix[row + nOfRows*col] += h;

	/* subtract h from each uncovered column */
	for(col=0; col<nOfColumns; col++)
		if(!coveredColumns[col])
			for(row=0; row<nOfRows; row++)
				distMatrix[row + nOfRows*col] -= h;

	/* move to step 3 */
	step3_dbl(assignment, distMatrix, starMatrix, newStarMatrix, primeMatrix, coveredColumns, coveredRows, nOfRows, nOfColumns, minDim);
}
