.PHONY: clean coverage develop env extras package release test virtualenv build_ext shell doc-html doc-sources doc-view doc-publish gh-pages

PYMODULE := clustering_metrics
PYPI_HOST := pypi
DISTRIBUTE := sdist bdist_wheel
SHELL_PRELOAD := $(PYMODULE)/_workspace.py

BUILD_STAMP = .build_stamp
ENV_STAMP = env/bin/activate
INTERPRETER := python3

SRC_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
SHELL_PRELOAD := $(SRC_ROOT)/$(SHELL_PRELOAD)

EXTENSION_PYX := $(shell find $(PYMODULE) -type f -name '*.pyx')
EXTENSION_PYX_MOD := $(patsubst %.pyx,%.cpp,$(EXTENSION_PYX))

EXTENSION_PYF := $(shell find $(PYMODULE) -type f -name '*.pyf')
EXTENSION_PYF_MOD := $(patsubst %.pyf,%module.c,$(EXTENSION_PYF))

EXTENSION_SO := \
	$(patsubst %.pyf,%.so,$(EXTENSION_PYF)) \
	$(patsubst %.cpp,%.so,$(EXTENSION_PYX_MOD))

EXTRAS_REQS := $(wildcard extras-*-requirements.txt)

PYENV := . env/bin/activate;
PYTHON := $(PYENV) python
PIP := $(PYENV) pip
HTML_DOCS := docs/_build/html

VENV_OPTS=""
ifeq ($(PIP_SYSTEM_SITE_PACKAGES),1)
VENV_OPTS="--system-site-packages"
endif

BROWSER := x-www-browser
ifeq ($(shell uname -s), Darwin)
BROWSER := open
endif

doc-sources:
	sphinx-apidoc \
		-A "`$(PYTHON) setup.py --author`" \
		-H "`$(PYTHON) setup.py --name`" \
		-V "`$(PYTHON) setup.py --version`" \
		-f -e -d 4 -F -o docs $(PYMODULE)
	-git checkout docs/conf.py
	-git checkout docs/Makefile
	-git add docs/index.rst
	-git commit -m"update doc sources"

doc-html: env build_ext
	make --directory=docs html

doc-view:
	$(BROWSER) $(HTML_DOCS)/index.html

doc-publish:
	git checkout --orphan gh-pages || git checkout gh-pages
	git reset
	find . -path ./.git -prune -o -path ./env -prune -o -path ./$(HTML_DOCS) -prune -o -type f -exec rm -f {} \;
	cp -R $(HTML_DOCS)/* .
	echo "" > .gitignore
	echo "docs/" >> .gitignore
	echo "env/" >> .gitignore
	touch .nojekyll
	-git add -A
	-git commit -a -m"update github pages"
	git push --set-upstream origin gh-pages || git push origin
	git checkout master

package: env build_ext
	$(PYTHON) setup.py $(DISTRIBUTE)

release: env build_ext
	$(PYTHON) setup.py $(DISTRIBUTE) upload -r $(PYPI_HOST)

coverage: test
	$(BROWSER) cover/index.html

test: env build_ext
	# make sure package can be pip-installed from local directory
	$(PIP) install -e .
	# run tests
	$(PYENV) pytest

shell: extras build_ext
	$(PYENV) PYTHONSTARTUP=$(SHELL_PRELOAD) ipython

extras: env/make.extras
env/make.extras: $(EXTRAS_REQS) | env
	$(PYENV) for req in $?; do pip install -r $$req; done
	touch $@

nuke: clean
	rm -rf *.egg *.egg-info env bin cover coverage.xml

clean:
	-python setup.py clean
	rm -rf dist build
	rm -f $(EXTENSION_SO) $(EXTENSION_PYF_MOD) $(EXTENSION_PYX_MOD)
	find . -path ./env -prune -o -type f -name "*.pyc" -exec rm -f {} \;

build_ext: env
	$(PYTHON) setup.py build_ext --inplace
	$(PYENV) find $(PYMODULE) -type f -name "setup.py" -exec python {} build_ext --inplace \;

$(EXTENSION_SO): build_ext
	@echo "done building $@"

develop: build_ext
	@echo "Installing for " `which pip`
	-pip uninstall --yes $(PYMODULE)
	pip install -e .

.PRECIOUS: $(ENV_STAMP)
.PHONY: env
env: $(ENV_STAMP)  ## set up a virtual environment
$(ENV_STAMP): setup.py requirements.txt
	test -f $@ || $(INTERPRETER) -m venv $(VENV_OPTS) env
	$(PIP) install -U pip wheel
	export SETUPTOOLS_USE_DISTUTILS=stdlib; $(PIP) install -r requirements.txt
	$(PIP) freeze > pip-freeze.txt
	$(PIP) install -e .
	touch $@
