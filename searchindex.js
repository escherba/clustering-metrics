Search.setIndex({envversion:49,filenames:["clustering_metrics","clustering_metrics.cluster","clustering_metrics.cluster_alt","clustering_metrics.entropy","clustering_metrics.ext","clustering_metrics.fent","clustering_metrics.fent.setup","clustering_metrics.fixes","clustering_metrics.hashes","clustering_metrics.hungarian","clustering_metrics.metrics","clustering_metrics.monte_carlo","clustering_metrics.monte_carlo.predictions","clustering_metrics.monte_carlo.strings","clustering_metrics.monte_carlo.utils","clustering_metrics.preprocess","clustering_metrics.ranking","clustering_metrics.utils","index"],objects:{"":{clustering_metrics:[0,0,0,"-"]},"clustering_metrics.entropy":{assignment_cost:[3,1,1,""],centropy:[3,1,1,""],cnum_pairs:[3,1,1,""],csum_pairs:[3,1,1,""],emi_from_margins:[3,1,1,""],fentropy:[3,1,1,""],fnum_pairs:[3,1,1,""],fsum_pairs:[3,1,1,""],lgamma:[3,1,1,""],ndarray_from_iter:[3,1,1,""]},"clustering_metrics.ext":{PHashCombiner:[4,2,1,""],VarlenHash:[4,2,1,""],hash_builtin_128:[4,1,1,""],hash_builtin_64:[4,1,1,""],hash_combine_boost:[4,1,1,""],hash_combine_boost_64:[4,1,1,""],hash_combine_murmur:[4,1,1,""],hash_combine_murmur_64:[4,1,1,""],hash_md5_128:[4,1,1,""],hash_md5_64:[4,1,1,""],hashable:[4,1,1,""],long2int:[4,1,1,""]},"clustering_metrics.ext.PHashCombiner":{combine:[4,3,1,""]},"clustering_metrics.fent":{setup:[6,0,0,"-"]},"clustering_metrics.fent.setup":{configuration:[6,1,1,""]},"clustering_metrics.fixes":{bincount:[7,1,1,""],isclose:[7,1,1,""]},"clustering_metrics.hungarian":{linear_sum_assignment:[9,1,1,""]},"clustering_metrics.metrics":{ClusteringMetrics:[10,2,1,""],ConfusionMatrix2:[10,2,1,""],ContingencyTable:[10,2,1,""],adjusted_mutual_info_score:[10,1,1,""],adjusted_rand_score:[10,1,1,""],cohen_kappa:[10,1,1,""],confmat2_type:[10,4,1,""],geometric_mean:[10,1,1,""],geometric_mean_weighted:[10,1,1,""],harmonic_mean:[10,1,1,""],harmonic_mean_weighted:[10,1,1,""],homogeneity_completeness_v_measure:[10,1,1,""],jaccard_similarity:[10,1,1,""],mutual_info_score:[10,1,1,""],product_moment:[10,1,1,""],ratio2weights:[10,1,1,""],unitsq_sigmoid:[10,1,1,""]},"clustering_metrics.metrics.ClusteringMetrics":{adjusted_fowlkes_mallows:[10,3,1,""],adjusted_rand_index:[10,3,1,""],fowlkes_mallows:[10,3,1,""],get_score:[10,3,1,""],mirkin_match_coeff:[10,3,1,""],mirkin_mismatch_coeff:[10,3,1,""],pairwise:[10,4,1,""],rand_index:[10,3,1,""]},"clustering_metrics.metrics.ConfusionMatrix2":{ACC:[10,3,1,""],DOR:[10,3,1,""],FDR:[10,3,1,""],FN:[10,4,1,""],FNR:[10,3,1,""],FOR:[10,3,1,""],FP:[10,4,1,""],FPR:[10,3,1,""],NLL:[10,3,1,""],NPV:[10,3,1,""],PLL:[10,3,1,""],PPV:[10,3,1,""],TN:[10,4,1,""],TNR:[10,3,1,""],TP:[10,4,1,""],TPR:[10,3,1,""],accuracy:[10,3,1,""],bias_index:[10,3,1,""],cole_coeff:[10,3,1,""],covar:[10,3,1,""],dice_coeff:[10,3,1,""],diseq_coeff:[10,3,1,""],frequency_bias:[10,3,1,""],from_ccw:[10,5,1,""],from_random_counts:[10,5,1,""],from_sets:[10,5,1,""],fscore:[10,3,1,""],get_score:[10,3,1,""],hypergeometric:[10,3,1,""],informedness:[10,3,1,""],jaccard_coeff:[10,3,1,""],kappa:[10,3,1,""],kappas:[10,3,1,""],lform:[10,3,1,""],loevinger_coeff:[10,3,1,""],markedness:[10,3,1,""],matthews_corr:[10,3,1,""],mic_scores:[10,3,1,""],mp_corr:[10,3,1,""],ochiai_coeff:[10,3,1,""],ochiai_coeff_adj:[10,3,1,""],overlap_coeff:[10,3,1,""],pairwise_hcv:[10,3,1,""],precision:[10,3,1,""],prevalence_index:[10,3,1,""],recall:[10,3,1,""],sensitivity:[10,3,1,""],sokal_sneath_coeff:[10,3,1,""],specificity:[10,3,1,""],to_ccw:[10,3,1,""],xcoeff:[10,3,1,""],yule_q:[10,3,1,""],yule_y:[10,3,1,""]},"clustering_metrics.metrics.ContingencyTable":{adjust_to_null:[10,3,1,""],adjusted_mutual_info:[10,3,1,""],assignment_score:[10,3,1,""],assignment_score_m1:[10,3,1,""],assignment_score_m2c:[10,3,1,""],assignment_score_m2r:[10,3,1,""],assignment_score_m3:[10,3,1,""],bc_metrics:[10,3,1,""],chisq_score:[10,3,1,""],col_diag:[10,3,1,""],entropy_scores:[10,3,1,""],expected:[10,3,1,""],expected_freqs_:[10,3,1,""],g_score:[10,3,1,""],muc_scores:[10,3,1,""],mutual_info_score:[10,3,1,""],row_diag:[10,3,1,""],split_join_distance:[10,3,1,""],split_join_similarity:[10,3,1,""],split_join_similarity_m1:[10,3,1,""],split_join_similarity_m2c:[10,3,1,""],split_join_similarity_m2r:[10,3,1,""],split_join_similarity_m3:[10,3,1,""],talburt_wang_index:[10,3,1,""],to_array:[10,3,1,""],vi_distance:[10,3,1,""],vi_similarity:[10,3,1,""],vi_similarity_m1:[10,3,1,""],vi_similarity_m2c:[10,3,1,""],vi_similarity_m2r:[10,3,1,""],vi_similarity_m3:[10,3,1,""]},"clustering_metrics.monte_carlo":{predictions:[12,0,0,"-"],utils:[14,0,0,"-"]},"clustering_metrics.monte_carlo.predictions":{Grid:[12,2,1,""],auc_xscaled:[12,1,1,""],create_plots:[12,1,1,""],do_mapper:[12,1,1,""],do_reducer:[12,1,1,""],get_conf:[12,1,1,""],join_clusters:[12,1,1,""],parse_args:[12,1,1,""],relabel_negatives:[12,1,1,""],run:[12,1,1,""],sample_with_error:[12,1,1,""],simulate_clustering:[12,1,1,""],simulate_labeling:[12,1,1,""],split_clusters:[12,1,1,""]},"clustering_metrics.monte_carlo.predictions.Grid":{best_clustering_by_score:[12,3,1,""],compare:[12,3,1,""],compute:[12,3,1,""],corrplot:[12,3,1,""],describe_matrices:[12,3,1,""],fill_clusters:[12,3,1,""],fill_matrices:[12,3,1,""],fill_sim_clusters:[12,3,1,""],find_highest:[12,3,1,""],find_matching_matrix:[12,3,1,""],iter_clusters:[12,3,1,""],iter_grid:[12,3,1,""],iter_matrices:[12,3,1,""],matrix_from_labels:[12,6,1,""],matrix_from_matrices:[12,6,1,""],plot:[12,6,1,""],show_cluster:[12,3,1,""],show_matrix:[12,3,1,""],with_clusters:[12,5,1,""],with_matrices:[12,5,1,""],with_sim_clusters:[12,5,1,""]},"clustering_metrics.monte_carlo.utils":{serialize_args:[14,1,1,""]},"clustering_metrics.ranking":{LiftCurve:[16,2,1,""],RocCurve:[16,2,1,""],aul_score_from_clusters:[16,1,1,""],aul_score_from_labels:[16,1,1,""],dist_auc:[16,1,1,""],num2bool:[16,1,1,""],roc_auc_score:[16,1,1,""]},"clustering_metrics.ranking.LiftCurve":{aul_score:[16,3,1,""],from_clusters:[16,5,1,""],from_counts:[16,5,1,""],from_labels:[16,5,1,""],plot:[16,3,1,""]},"clustering_metrics.ranking.RocCurve":{auc_score:[16,3,1,""],from_clusters:[16,5,1,""],from_labels:[16,5,1,""],from_scores:[16,5,1,""],max_informedness:[16,3,1,""],optimal_cutoff:[16,3,1,""],plot:[16,3,1,""]},"clustering_metrics.utils":{fill_with_last:[17,1,1,""],gapply:[17,1,1,""],get_df_subset:[17,1,1,""],getpropval:[17,1,1,""],lapply:[17,1,1,""],random_string:[17,1,1,""],randset:[17,1,1,""],sigsim:[17,1,1,""],sort_by_length:[17,1,1,""],tsorted:[17,1,1,""],wrap_scalar:[17,1,1,""]},clustering_metrics:{entropy:[3,0,0,"-"],ext:[4,0,0,"-"],fent:[5,0,0,"-"],fixes:[7,0,0,"-"],hungarian:[9,0,0,"-"],metrics:[10,0,0,"-"],monte_carlo:[11,0,0,"-"],ranking:[16,0,0,"-"],utils:[17,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","classmethod","Python class method"],"6":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute","5":"py:classmethod","6":"py:staticmethod"},terms:{"00001e10":7,"0001e10":7,"10th":10,"10x":3,"17th":10,"1e10":7,"26th":10,"2x2":10,"36th":10,"6th":10,"8th":16,"abstract":10,"amig\u00f3":10,"boolean":[7,9,16],"case":[3,7,9,10,16],"class":[3,4,10,12,16],"default":[10,16],"final":10,"float":[3,7,10,16,17],"function":[3,4,9,10,16,17],"int":[4,7,10,16,17],"long":4,"new":[7,9,10],"null":[10,12],"return":[4,7,9,10,12,16,17],"short":16,"static":12,"switch":[10,16],"true":[7,10,12,16,17],"while":10,_fent:5,abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz:17,aberdeen:10,about:10,abov:[5,7,10,16],absenc:[10,16],absolut:7,abuse:16,academ:10,acc:10,accept:10,accord:[10,16],account:10,accur:[10,16],accuraci:10,accurat:10,achiev:10,acm:[10,16],across:10,actual:[10,16],addit:[3,10],adjust:10,adjust_to_nul:10,adjusted_fowlkes_mallow:10,adjusted_mutual_info:10,adjusted_mutual_info_scor:10,adjusted_rand_index:10,adjusted_rand_scor:10,adjustmend:10,advanc:10,aero:5,affin:10,after:[10,16],against:[7,16],agreement:10,agricultur:10,alarm:10,albatineh:10,algebra:10,algorithm:[0,4,9,10],alia:10,all:[7,10,16],allclos:7,allow:10,almudevar:10,alpha:12,alphabet:17,also:[3,9,10,16],altern:10,although:10,alwai:[10,16],amax:7,ami:10,among:16,amount:16,analys:10,analysi:10,analyt:10,ani:10,annaler:10,annot:10,annual:[10,16],another:10,answer:16,anti:16,appear:[10,16],appli:[10,16,17],applic:[10,16],applicat:10,apply:17,approach:[10,16],appropri:16,approxim:10,arabi:10,arang:[7,9],arbitrari:4,archiv:5,area:10,arg:[10,12,14,17],argument:[16,17],ari:10,aris:10,arithmet:16,arrai:[3,7,9,10,16],array_lik:7,artifici:10,artile:10,arxiv:10,asarrai:3,assign:[9,10],assignemnt:9,assignment:[9,10],assignment_cost:3,assignment_scor:10,assignment_score_m1:10,assignment_score_m2c:10,assignment_score_m2r:10,assignment_score_m3:10,associ:10,associat:10,assum:[3,10,16],assume:[3,16],assumpt:[10,16],asymptot:[10,16],atol:7,attain:10,attribut:10,auc:[12,16],auc_scor:16,auc_xscal:12,augment:7,august:10,aul:16,aul_scor:16,aul_score_from_clust:16,aul_score_from_label:16,automat:10,averag:[10,16],axi:10,bad:16,bag:10,bagga:10,bailei:10,baldwin:10,base:[3,4,10,12,16],basi:[10,16],bc_metric:10,becaus:[10,16],becom:10,been:10,begin:16,behav:10,behavior:10,belong:[3,10,16],below:[10,16],ben:10,bend:16,beneath:10,berechnung:10,berlin:10,best:10,best_clustering_by_scor:12,beta:10,better:[4,10],between:[7,10,16,17],bia:10,bias:10,bias_index:10,bibe:10,bin:[7,16],binar:16,binari:[10,16],bincount:7,binom:3,binomi:3,bioengin:10,bioinformat:10,biolog:10,biotechnolog:10,bipartit:[9,10],bit:4,black:12,blog:[4,10],bool:[7,16,17],boost:4,borrow:3,boston:10,both:[7,10],bottom:[10,16],bound:10,brena:10,briggman:10,brusco:10,bsd:3,bug:5,bugaj:10,build:10,build_ext:5,built:4,burger:10,calcul:[3,10,16],call:[3,7,10,16],cam:5,campaign:16,can:[3,9,10,16],cancel:10,cancer:10,cannot:[7,10],cardin:10,carlo:10,carri:10,cast:[3,7],categori:10,ceil:[10,16],centropi:3,chain:10,chanc:10,chang:16,characterist:[10,16],che210d:5,check:16,chi:10,china:10,chisq_scor:10,choos:[3,10],chosen:10,chunk:7,clark:16,classic:9,classif:[10,16],classmethod:[10,12,16],claus:3,clear:10,clockwis:10,close:10,clusteringmetr:10,cnum_pair:3,code:[7,10,16],coeffici:[3,10,16],cohen:10,cohen_kappa:10,coincid:10,col_diag:10,col_ind:9,cole:10,cole_coeff:10,collabor:16,collect:[3,10,16,17],collig:10,colloc:10,color:12,column:[9,10],comb:3,combin:4,comlab:5,common:10,commonli:10,compar:[7,10,12],comparison:10,compil:5,complement:10,complet:[9,10,16],complex:7,compon:10,compound:10,compris:10,comput:[3,9,10,12],computation:10,compute_result:12,condit:10,confer:[10,16],confid:16,configur:6,confmat2_typ:10,confus:10,confusionmatrix2:10,connolli:10,consensu:10,consid:[7,16],consider:10,consist:[4,10,16],consortium:10,constraint:10,contain:[7,10,16],context:10,conting:10,contingencyt:10,contribut:16,control:10,conveni:[3,10],convers:[3,10],convert:[10,17],corefer:10,coreferenc:10,corner:16,correct:[10,16],correctli:16,correl:[10,16],correspond:[3,9,10,16],corrplot:12,cosin:10,cost:[9,10],cost_matrix:9,could:[10,16],count:[3,7,10,16],counter:10,counts_pr:16,counts_tru:16,cours:5,covar:10,covari:10,cover:16,cpad:10,creat:[3,4,10,16],create_plot:12,creativ:16,creator:16,criteria:10,critic:10,cross:10,crosstab:10,crude:10,csum_pair:3,cube:10,cumul:16,current:10,curv:[10,16],cutoff:16,data:[10,16],datafram:17,dataset:10,davi:10,decent:10,decompos:10,defin:[10,16],definit:[10,16],delta:10,deltap:10,denk:10,denomin:10,dens:10,depend:[7,10,16],der:10,deriv:10,descend:17,describ:[9,10],describe_matric:12,descript:10,desiderata:16,design:[10,16],desir:10,despit:10,detail:10,detect:10,determin:10,develop:10,diagnost:10,diagon:[10,16],diagram:16,dice:10,dice_coeff:10,dict:3,differ:[3,7,10,16],digit:7,dim:17,dimens:[7,17],dimension:[7,10],direct:[10,16],directli:10,disagre:10,discoveri:10,discret:10,discrimin:10,discuss:10,diseq_coeff:10,disequilibrium:10,disjoint:10,displai:16,dist_auc:16,distanc:10,distinct:10,distribut:[10,12,16],dither:12,diverg:10,divid:10,dna:10,do_mapp:12,do_reduc:12,doc:5,document:10,doe:10,doesn:10,domain:3,don:[10,16],done:10,dongen:10,dor:10,dordrecht:10,doswel:10,doubl:3,draw:[16,17],dtype:[7,10,12],dual:10,due:10,dun:10,duplic:17,dure:[10,16],each:[7,9,10,12,16],earli:16,easi:16,edit:5,edu:5,educat:10,effici:10,either:[10,16],elabor:10,electron:16,element:[7,10,17],embrecht:10,emi_from_margin:3,empir:10,empiric:10,emploi:16,end:16,engr:5,enough:16,ensur:10,entir:16,entiti:10,entri:[3,9,10,16],entropi:0,entropy_scor:10,environment:10,epp:10,equal:[7,9,10,16],equal_nan:7,equat:7,equival:[7,10],erfolg:10,error:[10,12],error_distribut:12,estim:10,estimat:10,eurasip:10,evalu:[10,16],evaluat:10,even:[3,10,16],event:10,everi:[3,9,10,16],evid:10,exactli:10,exampl:[4,7,9,10,16],except:[10,16,17],exclud:10,exhibit:10,exist:[10,16],expect:[3,10,16],expected_freqs_:10,expens:[10,16],experi:10,experiment:5,express:10,expression:10,ext:0,extend:[3,10,17],extent:10,extern:10,extrem:10,extrins:10,f2py:5,f90:5,factori:10,failur:[10,16],fall:16,fallout:10,fals:[7,10,12,16],famili:10,far:16,faster:3,fawcett:10,fdr:10,featur:[10,16],fent:0,fentropi:3,field:[10,17],file:[5,7],fill:[12,16],fill_clust:12,fill_matric:12,fill_sim_clust:12,fill_with_last:17,filter:16,find:[9,10,16],find_highest:12,find_matching_matrix:12,finit:7,first:[9,10],fix:0,flip_sign:12,float16:12,fnr:10,fnum_pair:3,follow:[3,7,10],forecast:10,form:[10,12,16],formal:[9,10],formul:10,formula:10,found:[7,10,16],fowlk:10,fowlkes_mallow:10,fpr:[10,16],freq:3,frequenc:[3,10],frequency_bia:10,from:[3,7,10,16,17],from_ccw:10,from_clust:[10,16],from_count:16,from_label:[10,16],from_partit:10,from_random_count:10,from_scor:16,from_set:10,fscore:10,fsum_pair:3,func:17,further:10,g_score:10,galpha:12,gamma:3,gammaln:3,gappli:17,gbeta:12,gene:10,gener:[9,10,16,17],genom:10,geografiska:10,geometr:10,geometric_mean:10,geometric_mean_weight:10,get:10,get_conf:12,get_df_subset:17,get_scor:10,getpropv:17,giant:16,gini:16,giurcaneanu:10,give:[7,9,10,16,17],given:[3,7,10,12,16,17],glue:5,goal:[9,16],gonzalo:10,good:[10,16],graham:10,grand:10,graph:[9,10],graphic:16,grid:12,ground:[10,16],group:[10,16],gusfield:10,gute:10,guyon:10,haenszel:10,half:10,hand:16,handl:3,hannssen:10,happen:10,hardli:10,harmon:10,harmonic_mean:10,harmonic_mean_weight:10,harold:9,hasenclev:10,hash:4,hash_builtin_128:4,hash_builtin_64:4,hash_combin:4,hash_combine_boost:4,hash_combine_boost_64:4,hash_combine_murmur:4,hash_combine_murmur_64:4,hash_md5_128:4,hash_md5_64:4,hashabl:4,have:[5,10,16],heavi:16,heidelberg:10,heidk:10,height:16,helmstaedt:10,help:16,henc:10,here:[3,10,16],hess:10,hidden:16,high:10,highli:10,hirschman:10,histogram:7,hit:10,hoffman:10,homogen:[10,16],homogeneity_completeness_v_measur:10,how:10,howev:[3,10],html:5,http:[4,5],hubert:10,human:16,humana:10,hungarian:0,hur:10,hypergeometr:10,icann:10,ident:10,idx:12,ieee:10,iff:[9,10],ignor:10,iii:10,imag:10,imperfect:[10,16],implement:[10,16],impli:16,implic:10,implicitli:10,improperli:10,improv:10,inc:10,includ:10,incomplet:10,increas:[10,12,16],independ:10,index:[7,10,18],indic:[9,10],individu:16,inequ:16,inf:10,infer:10,infin:10,info:5,inform:10,informat:[3,10],informed:[10,16],inner:10,inplac:5,input:[3,4,7,17],ins:10,inspir:[4,16],instanc:[3,9,10,17],instancemethod:17,instantiat:[10,16],instead:[7,10,16],integ:[3,4,7,17],intellig:10,intent:5,interest:[3,10],intern:10,internat:10,interpret:10,interrat:10,interv:[10,12],introduc:10,introduct:10,invers:[10,12],irrelev:10,is_class_po:16,isclos:7,isda:10,item:[10,17],iter:[3,17],iter_clust:12,iter_grid:12,iter_matric:12,iterabl:[10,16],iterable1:10,iterable2:10,iterat:3,jaccard:10,jaccard_coeff:10,jaccard_similar:10,jmartin:5,job:9,join:[10,12],join_clust:12,join_neg:12,joint:10,jone:10,journal:10,june:10,just:10,kao:10,kappa:10,keller:10,keyword:7,kluwer:10,known:[9,10],kolmogorov:16,kuhn:9,kuiper:10,kullback:10,kuo:10,kwarg:[10,12,17],label:[10,12,16],label_tru:16,labels_pr:[10,16],labels_tru:[10,16],languag:10,lappli:17,larg:[3,10,16],larger:[7,10],largest:[7,16],last:[7,17],learn:[3,7,10,16],least:[7,10],leav:10,left:16,leibler:10,len:17,length:[4,7,16,17],less:10,let:[9,16],letter:[10,17],level:10,lewontin:10,lform:10,lgamma:3,licens:3,lift:16,liftcurv:16,like:[3,10,16],likelihood:10,limit:10,line:7,linear:[9,10],linear_sum_assign:9,linguist:10,link:[5,10,16],linkag:10,list:[3,4,10,16,17],literatur:[10,16],loeving:10,loevinger_coeff:10,log:[3,10],logist:9,long2int:4,longer:7,look:[10,16],lossili:4,lot:10,low:10,lower:10,lst:17,machin:10,made:10,magatti:10,mai:[3,10],make:10,mallow:10,manag:10,mani:[10,16],mantel:10,map:[3,4],maqc:10,march:9,margin:[3,10],marked:10,marker:[10,12,16],markov:10,mason:10,match:[9,10,12],mathemat:10,matplotlib:16,matric:10,matrix:[9,10],matrix_from_label:12,matrix_from_matric:12,matter:10,matthew:10,matthews_corr:10,mattthew:10,max_class:12,max_count:12,max_informed:16,maximin:10,maximum:[10,16],maxwel:10,mcc:10,mean:[10,16],measur:[10,16],medic:10,meet:10,meila:10,member:10,memori:10,mention:10,messag:[10,16],meteorolog:10,method:[3,9,10,16],metric:0,mic0:10,mic1:10,mic_scor:10,michaelnielsen:4,microarrai:10,might:7,mihalko:10,mine:16,minim:[9,16],minimum:[7,9],minlength:7,minu:16,mirkin:10,mirkin_match_coeff:10,mirkin_mismatch_coeff:10,misc:3,mismatch:10,miss:10,model:10,modifi:5,moment:10,monoton:[10,16],mont:10,monte_carlo:0,more:[9,10,16],most:[7,9,10],mostli:16,mp_corr:10,muc:10,muc_scor:10,much:10,multivari:10,munkr:9,murmur:4,must:[10,12,16],mutual:[3,10],mutual_info_scor:10,mutut:10,n_cluster:16,n_sampl:16,name:10,nan:[7,16],natur:[3,10,16],naval:9,ncluster:12,ndarrai:7,ndarray_from_it:3,nearli:10,necessari:[7,10,16],need:[3,5,7,9,10],neg:[3,7,10,12,16],nei:10,neighbor:16,neither:10,network:10,neural:10,neurocomput:10,nevertheless:10,nextgenet:10,niewiadomska:10,ninth:10,nll:10,nlp:10,nmi:10,nomin:10,non:[3,7,10,16],none:[6,7,10,12,16],nonneg:7,nor:10,normal:[3,10,16],note:[3,4,5,7,9,10,16],novel:10,novemb:10,npv:10,null_distribut:12,num2bool:16,num:[4,16],number:[7,10,12,16,17],numer:[3,10],numpi:[3,5,7,9,10,12],obj:[12,17],object:[3,4,12,16,17],observ:10,obtain:[10,16],obviou:10,occur:10,occurr:[7,10],ochiai:10,ochiai_coeff:10,ochiai_coeff_adj:10,odd:10,off:10,often:10,omiss:10,one:10,onli:[3,10,16],oper:10,operat:[10,16],opposit:16,optim:[9,10],optimal:16,optimal_cutoff:16,option:[7,10,16],order:[10,16,17],orderedcrosstab:10,org:[4,5],origin:10,orthogon:10,other:[10,12,16],otherwis:[7,16],ought:10,our:16,out:[5,7,10],output:[4,5,7],outsid:16,over:[7,10],overal:10,overestim:16,overlap:10,overlap_coeff:10,overwrit:5,own:12,p_err:12,page:[9,18],pair:[3,10,12],pairwis:10,pairwise_hcv:10,panda:17,paper:10,paramet:[3,4,7,9,10,16,17],parent_packag:6,parse_arg:12,part:10,particular:10,particularli:10,partit:[9,10],path:16,pattern:10,paul:10,pdf:5,pearson:10,peculiar:10,penal:16,perfect:[10,16],perfectli:16,perform:[7,10],person:10,pervers:16,phashcombin:4,phi:10,pilgrim:9,pillin:10,place:[10,12,16],pll:10,plot:[10,12,16],point:[3,7,10,16],polynomi:4,poorest:16,poorli:10,popul:10,population_s:12,pos_label:16,pos_ratio:12,posit:[7,10,16],possibl:[3,7,10,16],post:10,power:10,ppv:10,practic:10,precis:[3,10],predic:16,predict:[0,10,11],prefer:10,prefigur:10,preprint:10,presenc:[10,16],present:16,preserv:10,press:10,preval:10,prevalence_index:10,princip:10,probabilist:10,probabl:[10,12],problem:[0,9,10],procedur:10,proceed:[10,16],process:[10,16],produc:10,product:[10,16],product_mo:10,profil:10,promot:16,properli:10,properti:[10,16,17],proport:[10,16],propos:10,provid:10,psycholog:10,psychometrika:10,pub:10,puriti:10,pyf:5,pymaptool:10,python9:5,python:[4,5],pythonfortran:5,quadrat:10,qualiti:10,quarterli:[9,10],r10:10,r11:10,r12:10,r13:10,r14:10,r15:10,r16:10,r18:10,r19:10,r20:10,r21:10,r22:10,r23:10,r24:10,r25:10,r26:10,r27:10,r28:10,r29:10,r30:10,r31:10,r32:10,r33:10,r34:10,r35:10,r36:10,r37:10,r38:10,r39:10,r40:10,r41:10,r42:10,r43:10,r44:10,rais:7,ramirez:10,ramo:10,rand:10,rand_index:10,random:[10,17],random_str:17,randset:17,rang:[4,10,16],rank:[0,10],rare:[7,10],rate:10,rater:10,rather:10,ratio2weight:10,ratio:10,reach:16,real:16,recal:10,recast:10,receiv:[10,16],recent:[7,10],recognit:10,rectangular:9,redraw:10,reduc:[4,12],refer:[0,9,10],regardless:10,regress:10,rel:[7,10],relabel_neg:12,relat:10,relationship:10,reli:[10,16],relianc:10,remain:[10,16],reorder:16,replac:[3,10,16],replic:10,report:16,repres:10,represent:16,reproduc:10,requir:[7,16],research:[9,10],resolut:10,resolv:10,resourc:10,respect:[10,16],respons:10,result:[3,7,10,16],retriev:10,revers:17,review:10,rewrit:16,richest:16,right:16,risk:10,roc:[10,16],roc_auc_scor:16,roccurv:16,rol:10,root:10,roughli:10,round:10,row:[9,10],row_diag:10,row_ind:9,royal:10,rpad:10,rtol:7,rule:16,run:12,ruzzo:10,rxc:[3,10],safe:7,sake:3,same:[7,10,16,17],sampl:[10,16,17],sample_rang:17,sample_s:12,sample_weight:16,sample_with_error:12,santo:10,save:16,save_to:[12,16],scalar:[3,7,17],scale:[10,12,16],scheme:10,scholz:10,scienc:10,scientif:10,scikit:[3,7,10,16],scipi:[3,5,10],score:[10,12,16],score_group:16,scores0:16,scores1:16,scores_neg:16,scores_po:16,scoring_method:[10,16],search:[10,18],second:[9,10],see:[4,7,10,16],seed:12,seen:[10,16],segment:10,select:10,semi:16,sensit:10,separ:16,septemb:16,seri:16,serialize_arg:14,servic:16,set1:10,set2:10,set:[9,10,16,17],setup:[0,5],seung:10,sever:10,shape:[7,9,10,16],shell:5,should:10,show_clust:12,show_matrix:12,show_progress:12,shown:16,siam:9,sibl:10,side:[10,16],sigmoid:10,signal:10,signatur:[5,17],signific:10,sigsim:17,similar:[10,16,17],similarli:10,simpl:[10,16],simpson:10,simulate_clust:12,simulate_label:12,sinc:[10,16],singl:[7,10,16],situat:[10,16],size:[7,10,12,16],skill:10,sklean:10,slide:5,slightli:10,sligtli:10,slower:3,small:7,smallest:16,smirnov:16,sneath:10,social:10,societi:10,soet:10,sokal:10,sokal_sneath_coeff:10,solut:[10,16],solv:[9,10],some:[4,7,10,16],sometim:10,somewhat:10,somewher:16,sort:[9,16,17],sort_by_length:17,sourc:[6,7,9,10,12,14,16,17],space:[10,16],spam:16,spammer:16,spars:[10,16],sparsiti:10,special:[3,10,16],specif:[10,16],specifi:[7,10,16,17],split:[10,12],split_clust:12,split_join:12,split_join_dist:10,split_join_similar:10,split_join_similarity_m1:10,split_join_similarity_m2c:10,split_join_similarity_m2r:10,split_join_similarity_m3:10,springer:10,squar:[9,10],stabil:10,stabl:10,standard:[3,10],stanford:5,statement:0,statist:[10,16],stdin:7,steinlei:10,stella:10,step:[5,16],stepwis:16,still:16,storag:10,store:10,str:[16,17],strang:10,strong:10,structur:10,studi:10,sturmwarnungsdienst:10,sub:10,subclass:10,subset:17,subtract:[10,16],success:10,sum:[3,7,9,10],summari:10,supervis:[10,16],supplement:10,supremum:16,surpris:10,symmetr:[7,10,12],synonym:10,system:10,szymkiewicz:10,tabl:[3,10],tabu:10,take:[10,16],talburt:10,talburt_wang_index:10,technic:10,ted:10,tend:10,term:10,terminolog:10,terribl:4,test:[7,10],than:[3,7,9,10,16],thei:[9,10,16],them:16,theoret:10,theori:10,therebi:10,therefor:16,thi:[3,7,9,10,16],those:[3,16],though:[7,10],thought:10,three:10,threshold:[10,16],through:16,thrown:16,thu:10,time:[10,17],titl:12,tnr:10,to_arrai:10,to_ccw:10,togeth:[4,7],toler:7,too:[10,16],top:16,top_path:6,topic:10,total:[10,16],toward:[10,16],tpr:[10,16],traceback:7,transform:10,transport:9,treat:16,treatment:16,trigger:16,truth:[10,16],tsort:17,tupl:[3,10,17],turaga:10,turn:10,twice:10,two:[3,4,7,10,16,17],type:[3,4,7,10,12,17],typeerror:7,typic:[7,10],ucsb:5,unadjust:10,unclust:16,uncorrect:10,und:10,undefin:10,under:[10,16],underestim:16,underli:10,understand:10,undesir:16,unequ:10,uniformli:10,uniqu:[7,10,16],unit:10,unitsq_sigmoid:10,universe_s:10,unix:5,unlike:[3,10],unnorm:10,upper:10,use:[3,4],useful:5,useless:10,user:[5,16],using:10,util:[0,11],utterli:10,valid:10,valu:[3,4,7,10,16,17],value_rang:17,valueerror:7,vanilla:4,variabl:[4,7,10,16],variant:[9,10],variat:10,varlenhash:4,vector:[3,4,10],verbatim:3,verdejo:10,veri:[7,10,16],versa:9,version:[7,9,10],vertex:9,vertic:16,vi_dist:10,vi_similar:10,vi_similarity_m1:10,vi_similarity_m2c:10,vi_similarity_m2r:10,vi_similarity_m3:10,vice:9,vilain:10,vinh:10,vol:10,volum:10,wai:[10,16],walk:16,wang:10,want:3,warn:16,warren:10,weather:10,weight:[7,9,10],well:10,were:[10,16],what:[3,16],when:[3,10,16],where:[5,7,9,10,16],whether:[7,16,17],which:[3,10,16],whissel:16,whose:16,width:16,wikipedia:[9,10,16],william:10,windstarkevorhersagen:10,wise:7,with_clust:12,with_matric:12,with_sim_clust:12,with_warn:[10,12],within:[7,10],without:10,word:[10,16],work:[10,16],worker:9,workshop:10,world:10,worst:10,would:[10,16],wrap_scalar:17,www:5,xcoeff:10,xiao:10,xlabel:12,xlim:12,y_score:16,y_true:16,yang:10,yate:10,yeung:10,ylabel:12,ylim:12,you:[3,5,10],youden:[10,16],your:10,yule:10,yule_i:10,yule_q:10,zero:[10,16]},titles:["clustering_metrics package","clustering_metrics.cluster module","clustering_metrics.cluster_alt module","clustering_metrics.entropy module","clustering_metrics.ext module","clustering_metrics.fent package","clustering_metrics.fent.setup module","clustering_metrics.fixes module","clustering_metrics.hashes module","clustering_metrics.hungarian module","clustering_metrics.metrics module","clustering_metrics.monte_carlo package","clustering_metrics.monte_carlo.predictions module","clustering_metrics.monte_carlo.strings module","clustering_metrics.monte_carlo.utils module","clustering_metrics.preprocess module","clustering_metrics.ranking module","clustering_metrics.utils module","Welcome to clustering-metrics&#8217;s documentation!"],titleterms:{algorithm:16,cluster:[1,18],cluster_alt:2,clustering_metr:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],content:[0,5,11],document:18,entropi:3,ext:4,fent:[5,6],fix:7,hash:8,hungarian:9,indice:18,metric:[10,18],modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],monte_carlo:[11,12,13,14],packag:[0,5,11],predict:12,preprocess:15,problem:16,rank:16,refer:16,setup:6,statement:16,string:13,submodul:[0,5,11],subpackag:0,tabl:18,util:[14,17],welcom:18}})