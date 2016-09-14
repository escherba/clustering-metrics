Search.setIndex({envversion:49,filenames:["clustering_metrics","clustering_metrics.entropy","clustering_metrics.ext","clustering_metrics.fent","clustering_metrics.fent.setup","clustering_metrics.fixes","clustering_metrics.hungarian","clustering_metrics.metrics","clustering_metrics.monte_carlo","clustering_metrics.monte_carlo.predictions","clustering_metrics.monte_carlo.utils","clustering_metrics.ranking","clustering_metrics.skutils","clustering_metrics.utils","index"],objects:{"":{clustering_metrics:[0,0,0,"-"]},"clustering_metrics.entropy":{assignment_cost:[1,1,1,""],centropy:[1,1,1,""],cnum_pairs:[1,1,1,""],csum_pairs:[1,1,1,""],emi_from_margins:[1,1,1,""],fentropy:[1,1,1,""],fnum_pairs:[1,1,1,""],fsum_pairs:[1,1,1,""],lgamma:[1,1,1,""],ndarray_from_iter:[1,1,1,""]},"clustering_metrics.ext":{PHashCombiner:[2,2,1,""],VarlenHash:[2,2,1,""],hash_builtin_128:[2,1,1,""],hash_builtin_64:[2,1,1,""],hash_combine_boost:[2,1,1,""],hash_combine_boost_64:[2,1,1,""],hash_combine_murmur:[2,1,1,""],hash_combine_murmur_64:[2,1,1,""],hash_md5_128:[2,1,1,""],hash_md5_64:[2,1,1,""],hashable:[2,1,1,""],long2int:[2,1,1,""]},"clustering_metrics.ext.PHashCombiner":{combine:[2,3,1,""]},"clustering_metrics.fent":{setup:[4,0,0,"-"]},"clustering_metrics.fent.setup":{configuration:[4,1,1,""]},"clustering_metrics.fixes":{bincount:[5,1,1,""],isclose:[5,1,1,""]},"clustering_metrics.hungarian":{linear_sum_assignment:[6,1,1,""]},"clustering_metrics.metrics":{ClusteringMetrics:[7,2,1,""],ConfusionMatrix2:[7,2,1,""],ContingencyTable:[7,2,1,""],adjusted_mutual_info_score:[7,1,1,""],adjusted_rand_score:[7,1,1,""],cohen_kappa:[7,1,1,""],confmat2_type:[7,4,1,""],geometric_mean:[7,1,1,""],geometric_mean_weighted:[7,1,1,""],harmonic_mean:[7,1,1,""],harmonic_mean_weighted:[7,1,1,""],homogeneity_completeness_v_measure:[7,1,1,""],jaccard_similarity:[7,1,1,""],mutual_info_score:[7,1,1,""],product_moment:[7,1,1,""],ratio2weights:[7,1,1,""],unitsq_sigmoid:[7,1,1,""]},"clustering_metrics.metrics.ClusteringMetrics":{adjusted_fowlkes_mallows:[7,3,1,""],adjusted_rand_index:[7,3,1,""],fowlkes_mallows:[7,3,1,""],get_score:[7,3,1,""],mirkin_match_coeff:[7,3,1,""],mirkin_mismatch_coeff:[7,3,1,""],pairwise:[7,4,1,""],rand_index:[7,3,1,""]},"clustering_metrics.metrics.ConfusionMatrix2":{ACC:[7,3,1,""],DOR:[7,3,1,""],FDR:[7,3,1,""],FN:[7,4,1,""],FNR:[7,3,1,""],FOR:[7,3,1,""],FP:[7,4,1,""],FPR:[7,3,1,""],NLL:[7,3,1,""],NPV:[7,3,1,""],PLL:[7,3,1,""],PPV:[7,3,1,""],TN:[7,4,1,""],TNR:[7,3,1,""],TP:[7,4,1,""],TPR:[7,3,1,""],accuracy:[7,3,1,""],bias_index:[7,3,1,""],cole_coeff:[7,3,1,""],covar:[7,3,1,""],dice_coeff:[7,3,1,""],diseq_coeff:[7,3,1,""],frequency_bias:[7,3,1,""],from_ccw:[7,5,1,""],from_random_counts:[7,5,1,""],from_sets:[7,5,1,""],fscore:[7,3,1,""],get_score:[7,3,1,""],hypergeometric:[7,3,1,""],informedness:[7,3,1,""],jaccard_coeff:[7,3,1,""],kappa:[7,3,1,""],kappas:[7,3,1,""],lform:[7,3,1,""],loevinger_coeff:[7,3,1,""],markedness:[7,3,1,""],matthews_corr:[7,3,1,""],mic_scores:[7,3,1,""],mp_corr:[7,3,1,""],ochiai_coeff:[7,3,1,""],ochiai_coeff_adj:[7,3,1,""],overlap_coeff:[7,3,1,""],pairwise_hcv:[7,3,1,""],precision:[7,3,1,""],prevalence_index:[7,3,1,""],recall:[7,3,1,""],sensitivity:[7,3,1,""],sokal_sneath_coeff:[7,3,1,""],specificity:[7,3,1,""],to_ccw:[7,3,1,""],xcoeff:[7,3,1,""],yule_q:[7,3,1,""],yule_y:[7,3,1,""]},"clustering_metrics.metrics.ContingencyTable":{adjust_to_null:[7,3,1,""],adjusted_mutual_info:[7,3,1,""],assignment_score:[7,3,1,""],assignment_score_m1:[7,3,1,""],assignment_score_m2c:[7,3,1,""],assignment_score_m2r:[7,3,1,""],assignment_score_m3:[7,3,1,""],bc_metrics:[7,3,1,""],chisq_score:[7,3,1,""],col_diag:[7,3,1,""],entropy_scores:[7,3,1,""],expected:[7,3,1,""],expected_freqs_:[7,3,1,""],g_score:[7,3,1,""],muc_scores:[7,3,1,""],mutual_info_score:[7,3,1,""],row_diag:[7,3,1,""],split_join_distance:[7,3,1,""],split_join_similarity:[7,3,1,""],split_join_similarity_m1:[7,3,1,""],split_join_similarity_m2c:[7,3,1,""],split_join_similarity_m2r:[7,3,1,""],split_join_similarity_m3:[7,3,1,""],talburt_wang_index:[7,3,1,""],to_array:[7,3,1,""],vi_distance:[7,3,1,""],vi_similarity:[7,3,1,""],vi_similarity_m1:[7,3,1,""],vi_similarity_m2c:[7,3,1,""],vi_similarity_m2r:[7,3,1,""],vi_similarity_m3:[7,3,1,""]},"clustering_metrics.monte_carlo":{predictions:[9,0,0,"-"],utils:[10,0,0,"-"]},"clustering_metrics.monte_carlo.predictions":{Grid:[9,2,1,""],auc_xscaled:[9,1,1,""],create_plots:[9,1,1,""],do_mapper:[9,1,1,""],do_reducer:[9,1,1,""],get_conf:[9,1,1,""],join_clusters:[9,1,1,""],parse_args:[9,1,1,""],relabel_negatives:[9,1,1,""],run:[9,1,1,""],sample_with_error:[9,1,1,""],simulate_clustering:[9,1,1,""],simulate_labeling:[9,1,1,""],split_clusters:[9,1,1,""]},"clustering_metrics.monte_carlo.predictions.Grid":{best_clustering_by_score:[9,3,1,""],compare:[9,3,1,""],compute:[9,3,1,""],corrplot:[9,3,1,""],describe_matrices:[9,3,1,""],fill_clusters:[9,3,1,""],fill_matrices:[9,3,1,""],fill_sim_clusters:[9,3,1,""],find_highest:[9,3,1,""],find_matching_matrix:[9,3,1,""],iter_clusters:[9,3,1,""],iter_grid:[9,3,1,""],iter_matrices:[9,3,1,""],matrix_from_labels:[9,6,1,""],matrix_from_matrices:[9,6,1,""],plot:[9,6,1,""],show_cluster:[9,3,1,""],show_matrix:[9,3,1,""],with_clusters:[9,5,1,""],with_matrices:[9,5,1,""],with_sim_clusters:[9,5,1,""]},"clustering_metrics.monte_carlo.utils":{serialize_args:[10,1,1,""]},"clustering_metrics.ranking":{LiftCurve:[11,2,1,""],RocCurve:[11,2,1,""],aul_score_from_clusters:[11,1,1,""],aul_score_from_labels:[11,1,1,""],dist_auc:[11,1,1,""],num2bool:[11,1,1,""],roc_auc_score:[11,1,1,""]},"clustering_metrics.ranking.LiftCurve":{aul_score:[11,3,1,""],from_clusters:[11,5,1,""],from_counts:[11,5,1,""],from_labels:[11,5,1,""],plot:[11,3,1,""]},"clustering_metrics.ranking.RocCurve":{auc_score:[11,3,1,""],from_clusters:[11,5,1,""],from_labels:[11,5,1,""],from_scores:[11,5,1,""],max_informedness:[11,3,1,""],optimal_cutoff:[11,3,1,""],plot:[11,3,1,""]},"clustering_metrics.skutils":{DataConversionWarning:[12,7,1,""],UndefinedMetricWarning:[12,7,1,""],assert_all_finite:[12,1,1,""],auc:[12,1,1,""],check_consistent_length:[12,1,1,""],column_or_1d:[12,1,1,""],roc_curve:[12,1,1,""],stable_cumsum:[12,1,1,""]},"clustering_metrics.utils":{fill_with_last:[13,1,1,""],gapply:[13,1,1,""],get_df_subset:[13,1,1,""],getpropval:[13,1,1,""],lapply:[13,1,1,""],random_string:[13,1,1,""],randset:[13,1,1,""],sigsim:[13,1,1,""],sort_by_length:[13,1,1,""],tsorted:[13,1,1,""],wrap_scalar:[13,1,1,""]},clustering_metrics:{entropy:[1,0,0,"-"],ext:[2,0,0,"-"],fent:[3,0,0,"-"],fixes:[5,0,0,"-"],hungarian:[6,0,0,"-"],metrics:[7,0,0,"-"],monte_carlo:[8,0,0,"-"],ranking:[11,0,0,"-"],skutils:[12,0,0,"-"],utils:[13,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","function","Python function"],"2":["py","class","Python class"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","classmethod","Python class method"],"6":["py","staticmethod","Python static method"],"7":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:function","2":"py:class","3":"py:method","4":"py:attribute","5":"py:classmethod","6":"py:staticmethod","7":"py:exception"},terms:{"00001e10":5,"0001e10":5,"10th":7,"10x":1,"17th":7,"1e10":5,"26th":7,"2x2":7,"36th":7,"6th":7,"8th":11,"abstract":7,"amig\u00f3":7,"boolean":[5,6,11,12],"case":[1,5,6,7,11,12],"class":[1,2,7,9,11,12],"default":[7,11,12],"final":[7,12],"float":[1,5,7,11,12,13],"function":[1,2,6,7,11,12,13],"import":12,"int":[2,5,7,11,12,13],"long":2,"new":[5,6,7],"null":[7,9],"return":[2,5,6,7,9,11,12,13],"short":11,"static":9,"switch":[7,11],"throw":12,"true":[5,7,9,11,12,13],"while":7,_fent:3,abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz:13,aberdeen:7,about:7,abov:[3,5,7,11],absenc:[7,11],absolut:[5,12],abuse:11,academ:7,acc:7,accept:7,accord:[7,11],account:7,accur:[7,11],accuraci:7,accurat:7,achiev:7,acm:[7,11],across:7,actual:[7,11],addit:[1,7],adjust:7,adjust_to_nul:7,adjusted_fowlkes_mallow:7,adjusted_mutual_info:7,adjusted_mutual_info_scor:7,adjusted_rand_index:7,adjusted_rand_scor:7,adjustmend:7,advanc:7,aero:3,affin:7,after:[7,11],against:[5,11],agreement:7,agricultur:7,alarm:7,albatineh:7,algebra:7,algorithm:[0,2,6,7],alia:7,all:[5,7,11,12],allclos:[5,12],allow:7,almudevar:7,alpha:9,alphabet:13,also:[1,6,7,11],altern:7,although:7,alwai:[7,11],amax:5,ami:7,among:11,amount:11,analys:7,analysi:7,analyt:7,ani:7,annaler:7,annot:7,annual:[7,11],another:7,answer:11,anti:11,appear:[7,11,12],appli:[7,11,13],applic:7,applicat:7,apply:13,approach:[7,11],appropri:11,approxim:7,arabi:7,arang:[5,6],arbitrari:2,arbitrarili:12,archiv:3,area:[7,12],arg:[7,9,10,13],argument:[11,13],ari:7,aris:7,arithmet:11,arr:12,arrai:[1,5,6,7,11,12],array_lik:5,artifici:7,artile:7,arxiv:7,asarrai:1,ascend:12,assert_all_finit:12,assign:[6,7],assignemnt:6,assignment:[6,7],assignment_cost:1,assignment_scor:7,assignment_score_m1:7,assignment_score_m2c:7,assignment_score_m2r:7,assignment_score_m3:7,associ:7,associat:7,assum:[1,7,11,12],assume:[1,11],assumpt:[7,11],asymptot:[7,11],atol:[5,12],attain:7,attribut:7,auc:[9,11,12],auc_scor:11,auc_xscal:9,augment:5,august:7,aul:11,aul_scor:11,aul_score_from_clust:11,aul_score_from_label:11,automat:7,averag:[7,11],avoid:12,axi:7,bad:11,bag:7,bagga:7,bailei:7,baldwin:7,base:[1,2,7,9,11,12],basi:[7,11],bc_metric:7,becaus:[7,11],becom:7,been:7,begin:11,behav:7,behavior:7,belong:[1,7,11],below:[7,11],ben:7,bend:11,beneath:7,berechnung:7,berlin:7,best:7,best_clustering_by_scor:9,beta:7,better:[2,7],between:[5,7,11,13],bia:7,bias:7,bias_index:7,bibe:7,bin:[5,11],binar:11,binari:[7,11,12],bincount:5,binom:1,binomi:1,bioengin:7,bioinformat:7,biolog:7,biotechnolog:7,bipartit:[6,7],bit:2,black:9,blog:[2,7],bool:[5,11,13],boost:2,borrow:1,boston:7,both:[5,7,12],bottom:[7,11],bound:7,brena:7,briggman:7,brusco:7,bsd:1,bug:3,bugaj:7,build:7,build_ext:3,built:2,burger:7,calcul:[1,7,11,12],call:[1,5,7,11],cam:3,campaign:11,can:[1,6,7,11,12],cancel:7,cancer:7,cannot:[5,7],cardin:7,carlo:7,carri:7,cast:[1,5],categori:7,ceil:[7,11],centropi:1,chain:7,chanc:7,chang:11,characterist:[7,11,12],che210d:3,check:[11,12],check_consistent_length:12,chi:7,china:7,chisq_scor:7,choos:[1,7],chosen:7,chunk:5,clark:11,classic:6,classif:[7,11,12],classifi:12,classmethod:[7,9,11],claus:1,clear:7,clockwis:7,close:7,cluster:[7,9,11],clusteringmetr:7,cnum_pair:1,code:[5,7,11,12],coeffici:[1,7,11],cohen:7,cohen_kappa:7,coincid:7,col_diag:7,col_ind:6,cole:7,cole_coeff:7,collabor:11,collect:[1,7,11,13],collig:7,colloc:7,color:9,column:[6,7,12],column_or_1d:12,comb:1,combin:2,comlab:3,common:7,commonli:7,compar:[5,7,9],comparison:7,compil:3,complement:7,complet:[6,7,11],complex:5,compon:7,compound:7,compris:7,comput:[1,6,7,9,12],computation:7,compute_result:9,condit:7,confer:[7,11],confid:[11,12],configur:4,confmat2_typ:7,confus:7,confusionmatrix2:7,connolli:7,consensu:7,consid:[5,11,12],consider:7,consist:[2,7,11,12],consortium:7,constraint:7,contain:[5,7,11,12],context:7,conting:7,contingencyt:7,contribut:11,control:[7,12],conveni:[1,7],convers:[1,7,12],convert:[7,13],coordin:12,corefer:7,coreferenc:7,corner:11,correct:[7,11],correctli:11,correl:[7,11],correspond:[1,6,7,11,12],corrplot:9,cosin:7,cost:[6,7],cost_matrix:6,could:[7,11],count:[1,5,7,11],counter:7,counts_pr:11,counts_tru:11,cours:3,covar:7,covari:7,cover:11,cpad:7,creat:[1,2,7,11,12],create_plot:9,creativ:11,creator:11,criteria:7,critic:7,cross:7,crosstab:7,crude:7,csum_pair:1,cube:7,cumsum:12,cumul:[11,12],current:7,curv:[7,11,12],cutoff:11,data:[7,11,12],dataconversionwarn:12,datafram:13,dataset:7,davi:7,decent:7,decis:12,decision_funct:12,decompos:7,decreas:12,defin:[7,11],definit:[7,11],delta:7,deltap:7,denk:7,denomin:7,dens:7,depend:[5,7,11,12],der:7,deriv:7,descend:13,describ:[6,7],describe_matric:9,descript:7,desiderata:11,design:7,desir:7,despit:7,detail:7,detect:7,determin:7,develop:7,diagnost:7,diagon:[7,11],diagram:11,dice:7,dice_coeff:7,dict:1,differ:[1,5,7,11,12],digit:5,dim:13,dimens:[5,12,13],dimension:[5,7],direct:[7,11],directli:7,disagre:7,discoveri:7,discret:7,discrimin:7,discuss:7,diseq_coeff:7,disequilibrium:7,disjoint:7,displai:[11,12],dist_auc:11,distanc:7,distinct:7,distribut:[7,9,11],dither:9,diverg:7,divid:7,dna:7,do_mapp:9,do_reduc:9,doc:3,document:7,doe:7,doesn:7,domain:1,don:[7,11],done:7,dongen:7,dor:7,dordrecht:7,doswel:7,doubl:1,draw:[11,13],drop:12,drop_intermedi:12,dtype:[5,7,9],dual:7,due:7,dun:7,duplic:13,dure:[7,11,12],each:[5,6,7,9,11],earli:11,edit:3,edu:3,educat:7,effici:7,either:[7,11,12],elabor:7,electron:11,element:[5,7,12,13],els:12,embrecht:7,emi_from_margin:1,empir:7,empiric:7,emploi:11,end:11,engr:3,enough:11,ensur:[7,12],entir:11,entiti:7,entri:[1,6,7,11,12],entropi:0,entropy_scor:7,environment:7,epp:7,equal:[5,6,7,11],equal_nan:5,equat:5,equival:[5,7],erfolg:7,error:[7,9,12],error_distribut:9,estim:[7,12],estimat:7,eurasip:7,evalu:[7,11],evaluat:7,even:[1,7,11],event:7,everi:[1,6,7,11],evid:7,exactli:7,exampl:[2,5,6,7,11,12],except:[7,11,12,13],exclud:7,exhibit:7,exist:[7,11],expect:[1,7,11],expected_freqs_:7,expens:[7,11],experi:7,experiment:3,explicitli:12,express:7,expression:7,ext:0,extend:[1,7,13],extent:7,extern:7,extrem:7,extrins:7,f2py:3,f90:3,factori:7,failur:[7,11],fall:11,fallout:7,fals:[5,7,9,11,12],famili:7,far:11,faster:1,fawcett:7,fdr:7,featur:[7,11],fent:0,fentropi:1,field:[7,13],file:[3,5],fill:[9,11],fill_clust:9,fill_matric:9,fill_sim_clust:9,fill_with_last:13,filter:11,find:[6,7,11],find_highest:9,find_matching_matrix:9,finit:5,first:[6,7,12],fix:0,flat:12,flip_sign:9,float16:9,fnr:7,fnum_pair:1,follow:[1,5,7],forecast:7,form:[7,9,11],formal:[6,7],formul:7,formula:7,found:[5,7,11],fowlk:7,fowlkes_mallow:7,fpr:[7,11,12],freq:1,frequenc:[1,7],frequency_bia:7,from:[1,5,7,11,12,13],from_ccw:7,from_clust:[7,11],from_count:11,from_label:[7,11],from_partit:7,from_random_count:7,from_scor:11,from_set:7,fscore:7,fsum_pair:1,func:13,further:7,g_score:7,galpha:9,gamma:1,gammaln:1,gappli:13,gbeta:9,gene:7,gener:[6,7,11,12,13],genom:7,geografiska:7,geometr:7,geometric_mean:7,geometric_mean_weight:7,get:7,get_conf:9,get_df_subset:13,get_scor:7,getpropv:13,giant:11,gini:11,giurcaneanu:7,give:[5,6,7,11,13],given:[1,5,7,9,11,12,13],glue:3,goal:[6,11],gonzalo:7,good:[7,11],graham:7,grand:7,graph:[6,7],graphic:11,grid:9,ground:[7,11],group:[7,11],guid:12,gusfield:7,gute:7,guyon:7,haenszel:7,half:7,hand:11,handl:1,hannssen:7,happen:[7,12],hardli:7,harmon:7,harmonic_mean:7,harmonic_mean_weight:7,harold:6,hasenclev:7,hash:2,hash_builtin_128:2,hash_builtin_64:2,hash_combin:2,hash_combine_boost:2,hash_combine_boost_64:2,hash_combine_murmur:2,hash_combine_murmur_64:2,hash_md5_128:2,hash_md5_64:2,hashabl:2,have:[3,7,11,12],heavi:11,heidelberg:7,heidk:7,height:11,helmstaedt:7,help:11,henc:7,here:[1,7,11],hess:7,hidden:11,high:[7,12],highli:7,hirschman:7,histogram:5,hit:7,hoffman:7,homogen:[7,11],homogeneity_completeness_v_measur:7,how:7,howev:[1,7,11],html:3,http:[2,3,12],hubert:7,human:11,humana:7,hungarian:0,hur:7,hypergeometr:7,icann:7,idea:11,ident:7,idx:9,ieee:7,iff:[6,7],ignor:7,iii:7,imag:7,imperfect:[7,11],implement:[7,11,12],impli:11,implic:7,implicit:12,implicitli:7,improperli:7,improv:7,inc:7,includ:7,incomplet:7,increas:[7,9,11,12],independ:7,index:[5,7,14],indic:[6,7],individu:11,inequ:11,inf:7,infer:7,infin:[7,12],info:3,inform:7,informat:[1,7],informed:[7,11],inner:7,inplac:3,input:[1,2,5,12,13],ins:7,inspir:2,instanc:[1,6,7,12,13],instancemethod:13,instantiat:[7,11],instead:[5,7,11],integ:[1,2,5,13],intellig:7,intent:3,interest:[1,7],intern:7,internat:7,interpret:7,interrat:7,interv:[7,9],introduc:7,introduct:7,invalid:12,invers:[7,9],irrelev:7,is_class_po:11,isclos:5,isda:7,item:[7,13],iter:[1,13],iter_clust:9,iter_grid:9,iter_matric:9,iterabl:[7,11],iterable1:7,iterable2:7,iterat:1,jaccard:7,jaccard_coeff:7,jaccard_similar:7,jmartin:3,job:6,join:[7,9],join_clust:9,join_neg:9,joint:7,jone:7,journal:7,june:7,just:7,kao:7,kappa:7,keller:7,keyword:5,kluwer:7,known:[6,7],kolmogorov:11,kuhn:6,kuiper:7,kullback:7,kuo:7,kwarg:[7,9,13],label:[7,9,11,12],label_tru:11,labels_pr:[7,11],labels_tru:[7,11],languag:7,lappli:13,larg:[1,7,11],larger:[5,7],largest:[5,11],last:[5,13],learn:[1,5,7,11,12],least:[5,7],leav:7,left:11,leibler:7,len:13,length:[2,5,11,12,13],less:7,let:[6,11],letter:[7,13],level:7,lewontin:7,lform:7,lgamma:1,licens:1,lift:11,liftcurv:11,lighter:12,like:[1,7,11,12],likelihood:7,limit:7,line:5,linear:[6,7],linear_sum_assign:6,linguist:7,link:[3,7,11],linkag:7,list:[1,2,7,11,12,13],literatur:[7,11],loeving:7,loevinger_coeff:7,log:[1,7],logist:6,long2int:2,longer:5,look:[7,11],lossili:2,lot:7,low:[7,12],lower:7,lst:13,machin:7,made:7,magatti:7,mai:[1,7],make:7,mallow:7,manag:7,mani:[7,11],mantel:7,map:[1,2],maqc:7,march:6,margin:[1,7],marked:7,marker:[7,9,11],markov:7,mason:7,match:[6,7,9,12],mathemat:7,matplotlib:11,matric:7,matrix:[6,7,12],matrix_from_label:9,matrix_from_matric:9,matter:7,matthew:7,matthews_corr:7,mattthew:7,max:12,max_class:9,max_count:9,max_informed:11,maximin:7,maximum:[7,11],maxwel:7,mcc:7,mean:[7,11],measur:[7,11,12],medic:7,meet:7,meila:7,member:7,memori:7,mention:7,messag:[7,11],meteorolog:7,method:[1,6,7,11],metric:0,mic0:7,mic1:7,mic_scor:7,michaelnielsen:2,microarrai:7,might:5,mihalko:7,mine:11,minim:[6,11],minimum:[5,6],minlength:5,minu:11,mirkin:7,mirkin_match_coeff:7,mirkin_mismatch_coeff:7,misc:1,mismatch:7,miss:7,model:7,modifi:3,moment:7,monoton:[7,11],mont:7,monte_carlo:0,more:[6,7,11,12],most:[5,6,7],mostli:11,motiv:0,mp_corr:7,muc:7,muc_scor:7,much:7,multivari:7,munkr:6,murmur:2,must:[7,9,11,12],mutual:[1,7],mutual_info_scor:7,mutut:7,n_cluster:11,n_sampl:[11,12],n_threshold:12,name:7,nan:[5,11,12],natur:[1,7,11],naval:6,ncluster:9,ndarrai:[5,12],ndarray_from_it:1,nearli:7,necessari:[5,7,11],need:[1,3,5,6,7],neg:[1,5,7,9,11,12],nei:7,neighbor:11,neither:7,network:7,neural:7,neurocomput:7,nevertheless:7,nextgenet:7,niewiadomska:7,ninth:7,nll:7,nlp:7,nmi:7,nomin:7,non:[1,5,7,11,12],none:[4,5,7,9,11,12],nonneg:5,nor:7,normal:[1,7,11],note:[1,2,3,5,6,7,11,12],novel:7,novemb:7,npv:7,null_distribut:9,num2bool:11,num:[2,11],number:[5,7,9,11,13],numer:[1,7],numpi:[1,3,5,6,7,9,12],obj:[9,13],object:[1,2,9,11,12,13],observ:7,obtain:[7,11],obviou:7,occur:7,occurr:[5,7],ochiai:7,ochiai_coeff:7,ochiai_coeff_adj:7,odd:7,off:7,often:7,omiss:7,one:7,onli:[1,7,11],oper:[7,12],operat:[7,11],opposit:11,optim:[6,7],optimal:11,optimal_cutoff:11,option:[5,7,11,12],order:[7,11,12,13],orderedcrosstab:7,org:[2,3,12],origin:[7,11],orthogon:7,other:[7,9,11,12],otherwis:[5,11],ought:7,our:11,out:[3,5,7],output:[2,3,5],outsid:11,over:[5,7],overal:7,overestim:11,overlap:7,overlap_coeff:7,overwrit:3,own:9,p_err:9,page:[6,14],pair:[1,7,9,12],pairwis:7,pairwise_hcv:7,panda:13,paper:7,paramet:[1,2,5,6,7,11,12,13],parent_packag:4,parse_arg:9,part:7,particular:7,particularli:7,partit:[6,7],past:12,path:11,pattern:7,paul:7,pdf:3,pearson:7,peculiar:7,penal:11,perfect:[7,11],perfectli:11,perform:[5,7],person:7,pervers:11,phashcombin:2,phi:7,pilgrim:6,pillin:7,place:[7,9,11],pll:7,plot:[7,9,11,12],point:[1,5,7,11,12],polynomi:2,poorest:11,poorli:7,popul:7,population_s:9,pos_label:[11,12],pos_ratio:9,posit:[5,7,11,12],possibl:[1,5,7,11],post:7,power:7,ppv:7,practic:7,precis:[1,7,12],precision_recall_curv:12,pred:12,predic:11,predict:[0,7,8],prefer:7,prefigur:7,preprint:7,presenc:[7,11],present:11,preserv:7,press:7,preval:7,prevalence_index:7,princip:7,probabilist:7,probabl:[7,9,12],problem:[6,7],procedur:7,proceed:[7,11],process:[7,11],produc:[7,11],product:[7,11],product_mo:7,profil:7,promot:11,properli:7,properti:[7,11,13],proport:[7,11],propos:7,provid:7,psycholog:7,psychometrika:7,pub:7,puriti:7,pyf:3,pymaptool:7,python9:3,python:[2,3],pythonfortran:3,quadrat:7,qualiti:7,quarterli:[6,7],r10:7,r11:7,r12:7,r13:7,r14:7,r15:7,r16:7,r18:7,r19:7,r20:7,r21:7,r22:7,r23:7,r24:7,r25:7,r26:7,r27:7,r28:7,r29:7,r30:7,r31:7,r32:7,r33:7,r34:7,r35:7,r36:7,r37:7,r38:7,r39:7,r40:7,r41:7,r42:7,r43:7,r44:7,r45:12,rais:[5,12],ramirez:7,ramo:7,rand:7,rand_index:7,random:[7,13],random_str:13,randset:13,rang:[2,7,11,12],rank:[0,7],rare:[5,7],rate:[7,12],rater:7,rather:7,ratio2weight:7,ratio:7,ravel:12,reach:11,read:12,real:11,recal:[7,12],recast:7,receiv:[7,11,12],receiver_operating_characterist:12,recent:[5,7],recognit:7,rectangular:6,redraw:7,reduc:[2,9],refer:[0,6,7],regardless:7,regress:7,rel:[5,7,12],relabel_neg:9,relat:7,relationship:7,reli:[7,11],relianc:7,remain:[7,11],reorder:[11,12],replac:[1,7,11],replic:7,report:11,repres:[7,12],represent:11,reproduc:7,requir:[5,11],research:[6,7],resolut:7,resolv:7,resourc:7,respect:[7,11],respons:7,restrict:12,result:[1,5,7,11,12],retriev:7,revers:[12,13],review:7,rewrit:11,richest:11,right:11,risk:7,roc:[7,11,12],roc_auc_scor:[11,12],roc_curv:12,roccurv:11,rol:7,root:7,roughli:7,round:7,row:[6,7],row_diag:7,row_ind:6,royal:7,rpad:7,rtol:[5,12],rule:[11,12],run:9,ruzzo:7,rxc:[1,7],safe:5,sake:1,same:[5,7,11,12,13],sampl:[7,11,12,13],sample_rang:13,sample_s:9,sample_weight:[11,12],sample_with_error:9,santo:7,save:11,save_to:[9,11],scalar:[1,5,13],scale:[7,9,11],scheme:7,scholz:7,scienc:7,scientif:7,scikit:[1,5,7,11,12],scipi:[1,3,7,12],score:[7,9,11,12],score_group:11,scores0:11,scores1:11,scores_neg:11,scores_po:11,scoring_method:[7,11],search:[7,14],second:[6,7],see:[2,5,7,12],seed:9,seen:[7,11],segment:7,select:7,semi:11,sensit:7,separ:11,septemb:11,seri:11,serialize_arg:10,servic:11,set1:7,set2:7,set:[6,7,11,12,13],setup:[0,3],seung:7,sever:7,shape:[5,6,7,11,12],shell:3,should:[7,12],show_clust:9,show_matrix:9,show_progress:9,shown:11,siam:6,sibl:7,side:[7,11],sigmoid:7,signal:7,signatur:[3,13],signific:7,sigsim:13,similar:[7,11,13],similarli:7,simpl:[7,11],simpson:7,simulate_clust:9,simulate_label:9,sinc:[7,11,12],singl:[5,7,11],situat:[7,11],size:[5,7,9,11],skill:7,sklean:7,sklearn:12,skutil:0,slide:3,slightli:7,sligtli:7,slower:1,small:5,smallest:11,smirnov:11,sneath:7,social:7,societi:7,soet:7,sokal:7,sokal_sneath_coeff:7,solut:[7,11],solv:[6,7],some:[2,5,7,11,12],sometim:7,somewhat:[7,11],somewher:11,sort:[6,11,12,13],sort_by_length:13,sourc:[4,5,6,7,9,10,11,12,13],space:[7,11],spam:11,spammer:11,spars:[7,11,12],sparsiti:7,special:[1,7,11],specif:[7,11],specifi:[5,7,11,13],split:[7,9],split_clust:9,split_join:9,split_join_dist:7,split_join_similar:7,split_join_similarity_m1:7,split_join_similarity_m2c:7,split_join_similarity_m2r:7,split_join_similarity_m3:7,springer:7,squar:[6,7],stabil:7,stabl:7,stable_cumsum:12,standard:[1,7],stanford:3,statist:[7,11],stdin:5,steinlei:7,stella:7,step:[3,11],stepwis:11,still:11,storag:7,store:7,str:[11,13],strang:7,string:13,strong:7,structur:7,studi:7,sturmwarnungsdienst:7,sub:7,subclass:7,suboptim:12,subset:13,subtract:[7,11],success:7,sum:[1,5,6,7,12],summari:7,supervis:[7,11],supplement:7,supremum:11,surpris:7,symmetr:[5,7,9],synonym:7,system:7,szymkiewicz:7,tabl:[1,7],tabu:7,take:[7,11],talburt:7,talburt_wang_index:7,target:12,task:12,technic:7,ted:7,tend:7,term:7,terminolog:7,terribl:2,test:[5,7],than:[1,5,6,7,11],thei:[6,7,11,12],them:[11,12],theoret:7,theori:7,therebi:7,therefor:11,thi:[1,5,6,7,11,12],those:[1,11],though:[5,7],thought:7,three:7,threshold:[7,11,12],through:11,thrown:11,thu:7,time:[7,13],titl:9,tnr:7,to_arrai:7,to_ccw:7,togeth:[2,5],toler:[5,12],too:[7,11],top:11,top_path:4,topic:7,total:[7,11],toward:[7,11],tpr:[7,11,12],traceback:5,transform:7,transport:6,trapezoid:12,treat:11,treatment:11,trigger:11,truth:[7,11],tsort:13,tupl:[1,7,12,13],turaga:7,turn:7,twice:7,two:[1,2,5,7,11,13],type:[1,2,5,7,9,13],typeerror:5,typic:[5,7],ucsb:3,unadjust:7,unclust:11,uncorrect:7,und:7,undefin:7,undefinedmetricwarn:12,under:[7,11,12],underestim:11,underli:7,understand:7,undesir:11,unequ:7,uniformli:7,uniqu:[5,7,11],unit:7,unitsq_sigmoid:7,universe_s:7,unix:3,unlike:[1,7],unnorm:7,upon:12,upper:7,use:[1,2,12],useful:3,useless:7,user:[3,11,12],userwarn:12,using:7,util:[0,8],utterli:7,valid:7,valu:[1,2,5,7,11,12,13],value_rang:13,valueerror:[5,12],vanilla:2,variabl:[2,5,7,11],variant:[6,7],variat:7,varlenhash:2,vector:[1,2,7],verbatim:[1,12],verdejo:7,veri:[5,7,11],versa:6,version:[5,6,7],versionad:12,vertex:6,vertic:11,vi_dist:7,vi_similar:7,vi_similarity_m1:7,vi_similarity_m2c:7,vi_similarity_m2r:7,vi_similarity_m3:7,vice:6,vilain:7,vinh:7,vol:7,volum:7,wai:[7,11],walk:11,wang:7,want:1,warn:[11,12],warren:7,weather:7,weight:[5,6,7,12],well:7,were:[7,11],what:[1,11],when:[1,7,11,12],where:[3,5,6,7,11],whether:[5,11,12,13],which:[1,7,11,12],whissel:11,whose:11,width:11,wiki:12,wikipedia:[6,7,11,12],william:7,windstarkevorhersagen:7,wise:5,with_clust:9,with_matric:9,with_sim_clust:9,with_warn:[7,9],within:[5,7],without:7,word:[7,11],work:[7,11],worker:6,workshop:7,world:7,worst:7,would:[7,11,12],wrap_scalar:13,wrong:12,www:3,xcoeff:7,xiao:7,xlabel:9,xlim:9,y_score:[11,12],y_true:[11,12],yang:7,yate:7,yeung:7,ylabel:9,ylim:9,you:[1,3,7],youden:[7,11],your:7,yule:7,yule_i:7,yule_q:7,zero:[7,11]},titles:["clustering_metrics package","clustering_metrics.entropy module","clustering_metrics.ext module","clustering_metrics.fent package","clustering_metrics.fent.setup module","clustering_metrics.fixes module","clustering_metrics.hungarian module","clustering_metrics.metrics module","clustering_metrics.monte_carlo package","clustering_metrics.monte_carlo.predictions module","clustering_metrics.monte_carlo.utils module","clustering_metrics.ranking module","clustering_metrics.skutils module","clustering_metrics.utils module","Welcome to clustering-metrics&#8217;s documentation!"],titleterms:{algorithm:11,cluster:14,clustering_metr:[0,1,2,3,4,5,6,7,8,9,10,11,12,13],content:[0,3,8],document:14,entropi:1,ext:2,fent:[3,4],fix:5,hungarian:6,indice:14,metric:[7,14],modul:[0,1,2,3,4,5,6,7,8,9,10,11,12,13],monte_carlo:[8,9,10],motiv:11,packag:[0,3,8],predict:9,rank:11,refer:11,setup:4,skutil:12,submodul:[0,3,8],subpackag:0,tabl:14,util:[10,13],welcom:14}})