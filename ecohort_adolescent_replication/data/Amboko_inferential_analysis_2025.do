**Demographic characteristics

*Categorical vars
foreach var of varlist marriedp educ_cat educ_level educ_level2 wealth_cat quintile region literacy_level  employment employ_status m1_trimester preg_intent  m1_dangersigns ANCdepression m1_overall_health m1_risk_health gravidity violence_abuse  FU_bloodtest FU_urinetest ultrasound_done anc4_visits FU_folic m3_del_HF delivery_privacy del_mistreatment m3_postnatal_checkup m1_ANC1visit_qual del_overall_qual m3_recommend_provider  {
	tab `var' age_cat_num, col chi2
	}

*South Africa
foreach var of varlist marriedp educ_cat educ_level tertile region literacy_level  employment employ_status m1_trimester preg_intent  m1_dangersigns ANCdepression m1_overall_health m1_risk_health gravidity violence_abuse  atleast1_FU_bloodtest atleast1_FU_urinetest atleast1_ultrasound anc4_visits FU_folic m3_del_HF delivery_privacy del_mistreatment m3_postnatal_checkup m1_ANC1visit_qual del_overall_qual m3_recommend_provider  {
	tab `var' age_cat_num, col chi2
	}
	
*Continuous covariates
foreach var of varlist social_support gravidity continuity_of_care_index FirstANC_completeness  firstANC_counsel_comp FU_ANCvisits_total anc_total immediate_pp_counsel_comp {
	summ `var' if age_group==1
    summ `var' if age_group==0 
	ttest `var', by(age_cat_num)

 }
 
 *Continuity of care
foreach var of varlist firstANCvisit_1trimester anc4_visits {
	tab `var' age_cat_num, col chi2 m
		}
	
ta m3_del_HF age_cat_num, col  chi2
	
ta m3_postnatal_checkup age_cat_num if m3_del_HF==1, col chi2 

ttest continuity_of_care_index, by(age_cat_num)


 *MNH content of care
 foreach var of varlist FirstANC_completeness  firstANC_counsel_comp  FU_ANCvisits_total anc_total {
	ttest `var', by(age_cat_num)

 }
 
foreach var of varlist FU_bloodtest FU_urinetest atleast1_foodsup ultrasound_done FU_folic atleast1_FP_counsel m3_del_HF{
	tab `var' age_cat_num, col chi2
	}

*Delivery care 
foreach var of varlist m3_vag_exam m3_vag_exam_per m3_vag_exam_privacy m3_req_pain_relief  m3_pain_relief delivery_privacy del_mistreatment m3_postnatal_checkup {
	tab `var' age_cat_num if m3_del_HF==1, col chi2 m
	}
	

*postnatal counselling
foreach var of varlist m3_postnatal_checkup m3_619a m3_619b m3_619c m3_619d m3_619e m3_619g m3_619h {
		ta `var' age_cat_num if m3_del_HF==1, col chi2 
} 

ttest immediate_pp_counsel_comp if m3_del_HF==1, by(age_cat_num)


*overall rating
*1st ANC quality rating
ta m1_ANC1visit_qual age_cat_num, col chi2


foreach var of varlist del_overall_qual m3_recommend_provider  {
	tab `var' age_cat_num if m3_del_HF==1 & country==3, col chi2
}

*pregnancy outcomes
foreach var of varlist c_section episiotomy_done prolonged_labor icu_adm blood_trans m3_fistula m3_severe_health atleast1_obs_outcome depression_pp{
		tab `var' age_cat_num if m3_del_HF==1, col chi2 m
		*svy:tab `var' age_cat_num if m3_baby_alive==1, col per
} 

*factor analysis
*Ethiopia
keep country site study_id facility_name*  del_mistreatment atleast1_obs_outcome age_group educ_cat wealth_cat m1_trimester  del_HF_ownership marriedp region literacy_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse FU_bloodtest FU_urinetest ultrasound_done FU_folic delivery_privacy del_HF_level m3_another_HF m1_enrollage social_support gravidity FirstANC_completeness firstANC_counsel_comp FU_ANCvisits_total anc_total m3_baby_alive m3_1005* adole_age del_overall_qual m3_recommend_provider m3_1004* m1_ANC1visit_qual age_cat_num age_group educ_level marriedp educ_cat educ_level tertile site region literacy_level  educ_level employment employ_status m1_trimester preg_intent m1_dangersigns ANCdepression m1_overall_health m1_risk_health  violence_abuse FU_bloodtest FU_urinetest ultrasound_done  FU_folic  delivery_privacy del_mistreatment social_support gravidity FirstANC_completeness  firstANC_counsel_comp FU_ANCvisits_total anc_total immediate_pp_counsel_comp m3_del_HF del_HF_ownership del_HF_level m3_another_HF m1_ANC1visit_qual del_overall_qual m3_recommend_provider firstANC_counsel_comp FirstANC_completeness del_overall_qual m3_recommend_provider m3_del_HF del_HF_ownership del_HF_level m1_ANC1visit_qual  atleast1_foodsup  FU_folic atleast1_FP_counsel marriedp educ_cat educ_level tertile region literacy_level  employment employ_status m1_trimester preg_intent  m1_dangersigns ANCdepression m1_overall_health m1_risk_health gravidity violence_abuse  anc4_visits FU_folic m3_del_HF delivery_privacy del_mistreatment m3_postnatal_checkup m1_ANC1visit_qual del_overall_qual m3_recommend_provider social_support gravidity FirstANC_completeness  firstANC_counsel_comp FU_ANCvisits_total anc_total immediate_pp_counsel_comp  firstANCvisit_1trimester anc4_visits m3_del_HF m3_postnatal_checkup FU_bloodtest FU_urinetest atleast1_foodsup ultrasound_done FU_folic atleast1_FP_counsel m3_del_HF m3_vag_exam m3_vag_exam_per m3_vag_exam_privacy m3_req_pain_relief  m3_pain_relief delivery_privacy del_mistreatment m3_postnatal_checkup del_overall_qual m3_recommend_provider m1_ANC1visit_qual m3_postnatal_checkup m3_619a m3_619b m3_619c m3_619d m3_619e m3_619g m3_619h c_section episiotomy_done prolonged_labor icu_adm blood_trans m3_fistula m3_severe_health atleast1_obs_outcome depression_pp lowBMI_MUAC quintile adole_age location


*Kenya
keep country site study_id facility_name*  del_mistreatment atleast1_obs_outcome age_group educ_cat wealth_cat m1_trimester  del_HF_ownership marriedp region literacy_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse FU_bloodtest FU_urinetest ultrasound_done FU_folic delivery_privacy del_HF_level m3_another_HF m1_enrollage social_support gravidity FirstANC_completeness firstANC_counsel_comp FU_ANCvisits_total anc_total m3_baby_alive m3_1005* adole_age del_overall_qual m3_recommend_provider m3_1004* m1_ANC1visit_qual age_cat_num age_group educ_level marriedp educ_cat educ_level tertile site region literacy_level  educ_level employment employ_status m1_trimester preg_intent m1_dangersigns ANCdepression m1_overall_health m1_risk_health  violence_abuse FU_bloodtest FU_urinetest ultrasound_done  FU_folic  delivery_privacy del_mistreatment social_support gravidity FirstANC_completeness  firstANC_counsel_comp FU_ANCvisits_total anc_total immediate_pp_counsel_comp m3_del_HF del_HF_ownership del_HF_level m3_another_HF m1_ANC1visit_qual del_overall_qual m3_recommend_provider firstANC_counsel_comp FirstANC_completeness del_overall_qual m3_recommend_provider m3_del_HF del_HF_ownership del_HF_level m1_ANC1visit_qual  atleast1_foodsup  FU_folic atleast1_FP_counsel marriedp educ_cat educ_level tertile region literacy_level  employment employ_status m1_trimester preg_intent  m1_dangersigns ANCdepression m1_overall_health m1_risk_health gravidity violence_abuse  anc4_visits FU_folic m3_del_HF delivery_privacy del_mistreatment m3_postnatal_checkup m1_ANC1visit_qual del_overall_qual m3_recommend_provider social_support gravidity FirstANC_completeness  firstANC_counsel_comp FU_ANCvisits_total anc_total immediate_pp_counsel_comp  firstANCvisit_1trimester anc4_visits m3_del_HF m3_postnatal_checkup FU_bloodtest FU_urinetest atleast1_foodsup ultrasound_done FU_folic atleast1_FP_counsel m3_del_HF m3_vag_exam m3_vag_exam_per m3_vag_exam_privacy m3_req_pain_relief  m3_pain_relief delivery_privacy del_mistreatment m3_postnatal_checkup del_overall_qual m3_recommend_provider m1_ANC1visit_qual m3_postnatal_checkup m3_619a m3_619b m3_619c m3_619d m3_619e m3_619g m3_619h c_section episiotomy_done prolonged_labor icu_adm blood_trans m3_fistula m3_severe_health atleast1_obs_outcome depression_pp lowBMI_MUAC quintile adole_age location

*South Africa
keep country site study_id facility_name*  del_mistreatment atleast1_obs_outcome age_group educ_cat wealth_cat m1_trimester  del_HF_ownership marriedp region literacy_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse FU_bloodtest FU_urinetest ultrasound_done FU_folic delivery_privacy del_HF_level m3_another_HF m1_enrollage social_support gravidity FirstANC_completeness firstANC_counsel_comp FU_ANCvisits_total anc_total m3_baby_alive m3_1005* adole_age del_overall_qual m3_recommend_provider m3_1004* m1_ANC1visit_qual age_cat_num age_group educ_level marriedp educ_cat educ_level tertile site region literacy_level  educ_level employment employ_status m1_trimester preg_intent m1_dangersigns ANCdepression m1_overall_health m1_risk_health  violence_abuse FU_bloodtest FU_urinetest ultrasound_done  FU_folic  delivery_privacy del_mistreatment social_support gravidity FirstANC_completeness  firstANC_counsel_comp FU_ANCvisits_total anc_total immediate_pp_counsel_comp m3_del_HF del_HF_ownership del_HF_level m3_another_HF m1_ANC1visit_qual del_overall_qual m3_recommend_provider firstANC_counsel_comp FirstANC_completeness del_overall_qual m3_recommend_provider m3_del_HF del_HF_ownership del_HF_level m1_ANC1visit_qual atleast1_FU_bloodtest atleast1_FU_urinetest atleast1_foodsup atleast1_ultrasound FU_folic atleast1_FP_counsel marriedp educ_cat educ_level tertile region literacy_level  employment employ_status m1_trimester preg_intent  m1_dangersigns ANCdepression m1_overall_health m1_risk_health gravidity violence_abuse  atleast1_FU_bloodtest atleast1_FU_urinetest atleast1_ultrasound anc4_visits FU_folic m3_del_HF delivery_privacy del_mistreatment m3_postnatal_checkup m1_ANC1visit_qual del_overall_qual m3_recommend_provider social_support gravidity FirstANC_completeness  firstANC_counsel_comp FU_ANCvisits_total anc_total immediate_pp_counsel_comp  firstANCvisit_1trimester anc4_visits m3_del_HF m3_postnatal_checkup FU_bloodtest FU_urinetest atleast1_foodsup ultrasound_done FU_folic atleast1_FP_counsel m3_del_HF m3_vag_exam m3_vag_exam_per m3_vag_exam_privacy m3_req_pain_relief  m3_pain_relief delivery_privacy del_mistreatment m3_postnatal_checkup del_overall_qual m3_recommend_provider m1_ANC1visit_qual m3_postnatal_checkup m3_619a m3_619b m3_619c m3_619d m3_619e m3_619g m3_619h c_section episiotomy_done prolonged_labor icu_adm blood_trans m3_fistula m3_severe_health atleast1_obs_outcome depression_pp lowBMI_MUAC tertile adole_age location


*missingness
mdesc country site adole_age wealth_cat m1_trimester del_HF_ownership marriedp literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse delivery_privacy del_HF_level m3_another_HF social_support gravidity anc4_visits  m1_ANC1visit_qual del_overall_qual m3_recommend_provider m3_baby_alive


*correlation
corr country marriedp educ_cat educ_level quintile site region literacy_level  educ_level employment employ_status m1_trimester preg_intent m1_dangersigns ANCdepression m1_overall_health m1_risk_health  violence_abuse FU_bloodtest FU_urinetest ultrasound_done anctotal_cat FU_folic anc4_visits delivery_privacy del_mistreatment social_support primi_gravida FirstANC_completeness  firstANC_counsel_comp immediate_pp_counsel_comp m3_del_HF del_HF_ownership del_HF_level m3_another_HF m1_ANC1visit_qual del_overall_qual m3_recommend_provider m1_ANC1visit_qual

*pregnancy outcomes
logistic  atleast1_obs_outcome site age_group i.wealth_cat i.m1_trimester del_HF_ownership marriedp  literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse FU_bloodtest FU_urinetest ultrasound_done FU_folic anc4_visits delivery_privacy del_HF_level m3_another_HF  social_support gravidity FirstANC_completeness firstANC_counsel_comp  m1_ANC1visit_qual del_overall_qual  m3_recommend_provider if m3_baby_alive==1 & age_cat_num==1 

*South Africa
logistic atleast1_obs_outcome site del_HF_level del_HF_ownership adole_age marriedp educ_level i.wealth_cat literacy_level employ_status gravidity i.m1_trimester preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse FU_bloodtest FU_urinetest ultrasound_done FU_folic anc4_visits delivery_privacy  m3_another_HF  social_support FirstANC_completeness firstANC_counsel_comp  m1_ANC1visit_qual del_overall_qual  m3_recommend_provider if m3_baby_alive==1 & age_cat_num==1 

atleast1_FU_bloodtest atleast1_FU_urinetest atleast1_foodsup atleast1_ultrasound FU_folic atleast1_FP_counsel

*significant factors Ethiopia all women
melogit atleast1_obs_outcome age_group ANCdepression m3_another_HF if m3_baby_alive==1 || site: , or

*significant factors Kenya all women
melogit atleast1_obs_outcome age_group employ_status m1_dangersigns m3_another_HF   if m3_baby_alive==1 || site: ,or

*pregnancy outcomes among adolescents
melogit atleast1_obs_outcome site i.adole_age i.wealth_cat i.m1_trimester i.anctotal_cat i.del_HF_ownership marriedp  literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse FU_bloodtest FU_urinetest ultrasound_done FU_folic anc4_visits delivery_privacy del_HF_level m3_another_HF  social_support primi_gravida FirstANC_completeness firstANC_counsel_comp   del_overall_qual m1_ANC1visit_qual if m3_baby_alive==1 & age_group==1 || country2:

*gen firstANC_trimester=

*mistreatment
melogit del_mistreatment age_group i.wealth_cat i.m1_trimester i.del_HF_ownership marriedp literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse delivery_privacy del_HF_level m3_another_HF social_support primi_gravida  anc4_visits  m1_ANC1visit_qual del_overall_qual m3_recommend_provider if m3_baby_alive==1 || site: 

*significant factors Ethiopia all women
melogit del_mistreatment age_group  m1_dangersigns  m1_risk_health violence_abuse delivery_privacy gravidity  del_overall_qual m3_recommend_provider if m3_baby_alive==1 || site: ,or

*significant factors Kenya all women
melogit del_mistreatment age_group  gravidity   del_overall_qual m3_recommend_provider if m3_baby_alive==1 || site:, or


logistic del_mistreatment country2 site i.adole_age i.wealth_cat i.m1_trimester i.del_HF_ownership marriedp literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse delivery_privacy del_HF_level m3_another_HF social_support gravidity anc4_visits  m1_ANC1visit_qual del_overall_qual m3_recommend_provider

*South Africa
logistic del_mistreatment site i.adole_age i.wealth_cat i.m1_trimester i.del_HF_ownership marriedp literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse delivery_privacy m3_another_HF social_support gravidity anc4_visits  m1_ANC1visit_qual del_overall_qual m3_recommend_provider

*South Africa
logistic atleast1_obs_outcome site i.adole_age i.wealth_cat i.m1_trimester i.anctotal_cat i.del_HF_ownership marriedp  literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse FU_bloodtest FU_urinetest ultrasound_done FU_folic anc4_visits delivery_privacy del_HF_level m3_another_HF  social_support primi_gravida FirstANC_completeness firstANC_counsel_comp   del_overall_qual m1_ANC1visit_qual if m3_baby_alive==1 & age_group==1

*Significant factors Kenya in adolescents
melogit del_mistreatment i.m1_trimester i.educ_level m1_dangersigns  ANCdepression   delivery_privacy if m3_baby_alive==1 & age_group==1|| site:

*Significant factors both countries in adolescents
melogit del_mistreatment i.del_HF_ownership   delivery_privacy del_overall_qual if m3_baby_alive==1 & age_group==1|| country2: 

*Continuity_of_care
melogit continuity_of_care i.wealth_cat i.m1_trimester marriedp  literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression  m1_overall_health social_support gravidity FirstANC_completeness firstANC_counsel_comp    m1_ANC1visit_qual if m3_baby_alive==1 & age_group==1 || site:

*Kenya
foreach var of varlist adole_age age_group wealth_cat m1_trimester anctotal_cat del_HF_ownership  {
    melogit atleast1_obs_outcome i.`var'  if m3_baby_alive==1 || site: 
}  


foreach var of varlist marriedp  literacy_level educ_level employ_status preg_intent m1_dangersigns  ANCdepression m1_risk_health m1_overall_health violence_abuse FU_bloodtest FU_urinetest ultrasound_done FU_folic anc4_visits delivery_privacy del_HF_level m3_another_HF social_support gravidity firstANC_counsel_comp FirstANC_completeness del_overall_qual FU_ANCvisits_total anc_total m1_ANC1visit_qual {
    ta `var' atleast1_obs_outcome, col
	melogit atleast1_obs_outcome `var'  if m3_baby_alive==1 || site: 
   melogit atleast1_obs_outcome `var'  if m3_baby_alive==1 || site: 

} 
 
