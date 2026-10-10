*******************************************************************************
"Data preparation and alysis for Objective I"
*******************************************************************************
**These mothers have incomplete data regarding the Obstetric history (m1_1001 to m1_1010)
*As a result, good to treat these data as an incomplete record and drop them from further analysis
    [71509/row-546, 251706/row-584
    211432/row-588, 241916/row-691
    160130/row-867, 190135/row-939]
*------------------------------------------------------------------------------
*Reliability and validity test and result interpretation
*------------------------------------------------------------------------------
*Item-retest correlation (internal consistency) shows how each item correlated with the sum of the remaining items (>0.2 is acceptable), 
*Item-test correlation shows how each item strongly correlated with the overall scale score (>0.5 is acceptable).
*average inter-item correlation: measures how strongly items are related to each other on average (the value b/n 0.15 to 0.5 is optimal).
*Cronbach's alpha: measures the internal consistency of the scale (0.6 to 0.7 is acceptable, but moderate internal consistency)

*Reliability for PHQ-9 tool

alpha m1_phq9a m1_phq9b m1_phq9c m1_phq9d m1_phq9e m1_phq9f m1_phq9g m1_phq9h ///
 m1_phq9i,item std //at baseline
alpha m2_205a_r1 m2_205b_r1 m2_205c_r1 m2_205d_r1 m2_205e_r1 m2_205f_r1 ///
m2_205g_r1 m2_205h_r1 m2_205i_r1 ,item std // Survey two

alpha m2_205a_r2 m2_205b_r2 m2_205c_r2 m2_205d_r2 m2_205e_r2 m2_205f_r2 ///
m2_205g_r2 m2_205h_r2 m2_205i_r2 ,item std // survey three
alpha m2_205a_r3 m2_205b_r3 m2_205c_r3 m2_205d_r3 m2_205e_r3 m2_205f_r3 ///
m2_205g_r3 m2_205h_r3 m2_205i_r3 ,item std // Survey four

alpha m2_205a_r4 m2_205b_r4 m2_205c_r4 m2_205d_r4 m2_205e_r4 m2_205f_r4 ///
m2_205g_r4 m2_205h_r4 m2_205i_r4 ,item std // Survey five
alpha m2_205a_r5 m2_205b_r5 m2_205c_r5 m2_205d_r5 m2_205e_r5 m2_205f_r5 ///
m2_205g_r5 m2_205h_r5 m2_205i_r5 ,item std // survey six

alpha m2_205a_r6 m2_205b_r6 m2_205c_r6 m2_205d_r6 m2_205e_r6 m2_205f_r6 ///
m2_205g_r6 m2_205h_r6 m2_205i_r6 ,item std // Survey seven	
     * Allmeasurment point chronach alhp was>0.75, strongly acceptable
	 
*Relability Women HRQoL 
alpha m1_205a m1_205b m1_205c m1_205d m1_205e , item std //chrobach alph=0.54 & interitem correlation=0.19 during PX 
alpha m5_health_a m5_health_b m5_health_c m5_health_d m5_health_e, ///
item std // chrobach alph=0.64 & average inter-item correlation=0.0.26 and item-rest correlatin(>0.2) during ppp 

******************************************************************************
 "Missingness of data checking"
****************************************************************************
*At baseline (n=933)
preserve
drop if depressions2==.
misstable summarize (m1_phq9a m1_phq9b m1_phq9c m1_phq9d m1_phq9e m1_phq9f m1_phq9g m1_phq9h m1_207) 
misstable summarize ( m1_8a_et m1_8b_et m1_8c_et m1_8e_et m1_8f_et) // obstetric problems m1_8d_et(159 miss)
misstable summarize ( m1_205a m1_205b m1_205c m1_205d m1_205e) // Women HRQol
misstable summarize ( m1_813a m1_813b m1_813c m1_813d) // common health problems
misstable summarize ( m1_814a m1_814b m1_814c m1_814d m1_814e m1_814f m1_814g m1_814h m1_814i) // Px danger signs (m1_814h/fetus not moving (884 miss))
misstable summarize (m1_901 m1_903 m1_905) //smoke, khat and alcohol

misstable summarize (m1_muac m1_height_cm m1_weight_kg m1_Hb m1_bp_time_1_systolic ///
m1_bp_time_1_diastolic m1_bp_time_2_systolic m1_bp_time_2_diastolic ///
m1_bp_time_3_systolic m1_bp_time_3_diastolic m1_ga m1_trimester) // biomarker (anemia showed 5.5%) 

misstable summarize (m1_700 m1_701 m1_702 m1_703 m1_704 m1_705 m1_706 m1_707 m1_708a m1_710a m1_711a m1_714a m1_716b m1_716c m1_716e m1_724a m1_801 m1_806 m1_807 m1_1001) // m1_710a/syphills and m1_711a/DM test (159 vs 203 miss)
restore
*-----------------------------------------------------------------
*PHQ-9, PX dangers, Obstetric, WHRQoL, Substance use, IPV 
preserve
keep if m2_202_r1==1
misstable summarize m1_phq9a m1_phq9b m1_phq9c m1_phq9d m1_phq9e m1_phq9f m1_phq9g m1_phq9h m1_814a m1_814b m1_814c m1_814d m1_814e m1_814f m1_814g m1_814i  m1_8a_et m1_8b_et m1_8c_et m1_8e_et m1_8f_et m1_205a m1_205b m1_205c m1_205d m1_205e m1_muac m1_height_cm m1_weight_kg m1_Hb m1_ga m1_700 m1_701 m1_702 m1_703 m1_704 
restore

preserve
keep if m2_202_r1==1
misstable summarize m1_705 m1_706 m1_707 m1_708a m1_714a m1_716b m1_716c m1_716e m1_724a m1_801 m1_806 m1_807 m1_901 m1_903 m1_905 m1_813a m1_813b m1_813c m1_813d m1_1101_phy_viol m1_1103_emo_viol m1_1001
restore
*------------------------------------------------------------------------------
  "Missingness of data at module two (m2)"
*------------------------------------------------------------------------------
*At round one (m2_r1) (n=933); including, 
*ANDs, Px danger sign & substance use
preserve
keep if m2_202_r1==1
misstable summarize m2_203a_r1 m2_203b_r1 m2_203c_r1 m2_203d_r1 m2_203e_r1 m2_203f_r1 m2_203g_r1 m2_203h_r1 m2_203i_r1  m2_205a_r1 m2_205b_r1 m2_205c_r1 m2_205d_r1 m2_205e_r1 m2_205f_r1 m2_205g_r1 m2_205h_r1 m2_205i_r1 m2_206_r1 m2_207_r1 m2_208_r1 
// new health care consultation
 // checking for missing data
restore //0.3% missingness

*At round two(m2_r2) (n-855)
preserve
keep if m2_202_r2==1
misstable summarize m2_203a_r2 m2_203b_r2 m2_203c_r2 m2_203d_r2 m2_203e_r2 m2_203f_r2 m2_203g_r2 m2_203h_r2 m2_203i_r2 m2_204a_et_r2 m2_204b_et_r2 m2_204c_et_r2 m2_204d_et_r2 m2_204e_et_r2 m2_204f_et_r2 m2_204g_et_r2 m2_204h_et_r2 m2_204i_r2 m2_205a_r2 m2_205b_r2 m2_205c_r2 m2_205d_r2 m2_205e_r2 m2_205f_r2 m2_205g_r2 m2_205h_r2 m2_205i_r2 m2_206_r2 m2_207_r2 m2_208_r2 m2_301_r2 // new health care consultation
 // checking for missing data
restore // 0% missingness

*At round three(m2_r3) (n-728)
preserve
keep if m2_202_r3==1
misstable summ m2_203a_r3 m2_203b_r3 m2_203c_r3 m2_203d_r3 m2_203e_r3 m2_203f_r3 m2_203g_r3 m2_203h_r3 m2_203i_r3 m2_204a_et_r3 m2_204b_et_r3 m2_204c_et_r3 m2_204d_et_r3 m2_204e_et_r3 m2_204f_et_r3 m2_204g_et_r3 m2_204h_et_r3 m2_204i_r3 m2_205a_r3 m2_205b_r3 m2_205c_r3 m2_205d_r3 m2_205e_r3 m2_205f_r3 m2_205g_r3 m2_205h_r3 m2_205i_r3 m2_206_r3 m2_207_r3 m2_208_r3 m2_301_r3 // new health care consultation
 // checking for missing data
restore // 1% missingness

*At round four (n=506)
preserve
keep if m2_202_r4==1
misstable summ m2_203a_r4 m2_203b_r4 m2_203c_r4 m2_203d_r4 m2_203e_r4 m2_203f_r4 m2_203g_r4 m2_203h_r4 m2_203i_r4 m2_204a_et_r4 m2_204b_et_r4 m2_204c_et_r4 m2_204d_et_r4 m2_204e_et_r4 m2_204f_et_r4 m2_204g_et_r4 m2_204h_et_r4 m2_204i_r4 m2_205a_r4 m2_205b_r4 m2_205c_r4 m2_205d_r4 m2_205e_r4 m2_205f_r4 m2_205g_r4 m2_205h_r4 m2_205i_r4 m2_206_r4 m2_207_r4 m2_208_r4 m2_301_r4 // new health care consultation
 // checking for missing data
restore // 0% missingness

*At round 5 (N=240)
preserve
keep if m2_202_r5==1
misstable summ m2_203a_r5 m2_203b_r5 m2_203c_r5 m2_203d_r5 m2_203e_r5 m2_203f_r5 m2_203g_r5 m2_203h_r5 m2_203i_r5 m2_204a_et_r5 m2_204b_et_r5 m2_204c_et_r5 m2_204d_et_r5 m2_204e_et_r5 m2_204f_et_r5 m2_204g_et_r5 m2_204h_et_r5 m2_204i_r5 m2_205a_r5 m2_205b_r5 m2_205c_r5 m2_205d_r5 m2_205e_r5 m2_205f_r5 m2_205g_r5 m2_205h_r5 m2_205i_r5 m2_206_r5 m2_207_r5 m2_208_r5 m2_301_r5 // new health care consultation
 // checking for missing data
restore // 0% missingness

*At round 6 (N=85)
preserve
keep if m2_202_r6==1
misstable summ m2_203a_r6 m2_203b_r6 m2_203c_r6 m2_203d_r6 m2_203e_r6 m2_203f_r6 m2_203g_r6 m2_203h_r6 m2_203i_r6 m2_204a_et_r6 m2_204b_et_r6 m2_204c_et_r6 m2_204d_et_r6 m2_204e_et_r6 m2_204f_et_r6 m2_204g_et_r6 m2_204h_et_r6 m2_204i_r6 m2_205a_r6 m2_205b_r6 m2_205c_r6 m2_205d_r6 m2_205e_r6 m2_205f_r6 m2_205g_r6 m2_205h_r6 m2_205i_r6 m2_206_r6 m2_207_r6 m2_208_r6 m2_301_r6 // new health care consultation
 // checking for missing data
restore // 0% missingness

*-----------------------------------------------------------------------------   
 "To show the label of each choice, you can run this command"
*-----------------------------------------------------------------------------
numlabel, add
	
*To order more than two variables: example to order residence, next marital status, next occupation
order residence marital status, before (occupation) 
recode health_cat (1=5) (2=4) (3=3) (4=2) (5=1), gen(health_score)
  
*Rename multiple variable at once
rename (pxdangers1_cat  pxdangers2_cat) (pxdanger_cat1 pxdanger_cat2)	
rename (deprs1_5cutoff deprs2_5cutoff ) (depre_cat1 depre_cat2)
rename (deprs1_8cutoff deprs2_8cutoff ) (depre8cut_cat1 depre8cut_cat2)

********************************************************************************
 "Socio-demographic characteristics of the pregnant women"
********************************************************************************
*Women's educational level (m1_502 and m1_503)***
gen m1_edulevel_3cat=m1_503
recode m1_edulevel_3cat  (. 1 =1) (2/3=2)(4/5=3) 
lab var m1_edulevel_3cat "503. Educational level of women, 3 categories"
lab def m1_edulevel_3cat 1 "No education or some primary" 2 "Complete primary or some secondary" ///
 3 "Complete secondary or higher"
lab val m1_edulevel_3cat m1_edulevel_3cat
tab m1_edulevel_3cat
	 
**Women's health literacy*
  // Women's health literacy was assessed by using the six self-reported questions, and 
  // those who correctly answered were considered to have good health literacy
  // these questions were: m1_509a==1/Yes 0/no; m1_509b==1/Yes 0/no; 
  // m1_510a/head about illness of TB==1/Yes 0/no; 
  // m1_510b_TB treated using herbal or traditional medicine==1/Yes 0/no; 
  // m1_511/When children have diarrhea, 
  // m1_512/Is smoke from a wood-burning traditional stove good for health,

gen m1_health_literacy_2=m1_health_literacy
recode m1_health_literacy_2 (1/3=0) (4=1)
lab var m1_health_literacy_2 "Health literacy of women" // Correctly responding to all six health literacy questions
lab def m1_health_literacy_2 1 "Good" 0 "Poor"
lab val m1_health_literacy_2 m1_health_literacy_2
tab m1_health_literacy_2
	
*Age of the the mothers
gen m1_age_cat1=m1_enrollage
recode m1_age_cat1 (min/19=1) (20/34=2) (35/max=3)
lab var m1_age_cat1 "B3. Women's age in year"
lab def m1_age_cat1 1 "< 20 years" 2 "20-34 years" 3 "≥ 35 years"
lab val m1_age_cat1 m1_age_cat1
tab m1_age_cat1
	 
*For generating mean standard deviation and IQR of respondent agr
summarize m1_enrollage, detail //for generating IQR    IQR: 6(22-28 Years)  25.42(4.83)
sktest  m1_enrollage  // For checking Skewness & kurtosis tests for normality (Skewness=0 or kurtosis=3 show normal distribution) or 
	                  // Their Skewness & kurtosis tests of p-value>0.05 indicates normally distributed					 
*Residence (m1_residence)
gen m1_residence=study_site
recode m1_residence (1=1) (2/7=2)
lab var m1_residence "Residence of the respondent"
lab def m1_residence 1 "Urban" 2 "Rural"
lab val m1_residence m1_residence
tab m1_residence 
	  
*Marital status (m1_marital)
gen m1_marital=m1_505
recode m1_marital (1/4=0) (5/6=1)
lab var m1_marital "505. Marital status"
lab def m1_marital 0 "Not currently married"  1 "Currently married" 
lab val m1_marital m1_marital
tab m1_marital // mother with age 35 with id 231625/r-831 updated as married instead of missed. 

*Occupation (m1_occupation)  
gen m1_occupation= m1_506  
recode m1_occupation (1/3=1) (4=2) (6=3) (5 7/9 96=4 )
lab var m1_occupation "506. Occupation of the mother"
lab def m1_occupation 1 "Employed"  2 "Merchant" 3 "Housewife" 4 "Others"
lab val m1_occupation m1_occupation 
tab m1_occupation 
  // mother with 22232/row 954, her job is coded as "other", but she has her own pharmacy/clinic. 
  // As a result, I corrected this mother's occupation as "4.Merchant"
  // mother with id 241913/r-598 updated as other occupation instead of missed.
  *Employed [includes Public, Private and NGO]  
  *Other [students, labour, pastoralist, no jobs, handicraft, religious work] 
		   
*Religion (m1_religion)
gen m1_religion= m1_507  
recode m1_religion (1/3=1) (4=2) (5 96=3 )
lab var m1_religion "507. Religion of the mother"
lab def m1_religion 1 "Christian"  2 "Muslim" 3 "Others"
lab val m1_religion m1_religion
tab m1_religion
  *Christian [Orthodox, Catholic, Protestant] *Others: [Adventist/Wakefeta/Indigenous, Hawariyat]
  
*Number of currently alive children 
tab m1_num_alivechild // generated from women's obstetric history

*Economic status (wealth index, tertile): poorest, second, and richest*
 tab tertile 
 // should describe well the list of variables used to generate the wealth index and how to generate have?
 // have 16 missed data, so please check again the wealth index score of did agin the PCA: 
 // 18 women with the following id having missed wealth index scor
   * 211422, 10332, 301610,190138,  71513, 180824, 180804, 160121, 71512, 
   * 111302,50933, 151203,71527,180823, 271428, 71507, 160130, 211416
   
*Health insurance coverage
 tab m1_1221  
 // two mother had missed(281915/row-704 and 111318/row-835). 
 // However, the missed data was updated by checking the response of m1_1222
	   
*Type of Health Insurance   
 tab m1_1222	
	   
*For checking the right trimester level the HCWs provided the EDD for women(m1_801), 
// The EDD of the women provided by HCWs/ANC provider(m1_802a), 
// do you know LMP (m1_802b_et) what is lmp (m1_802c_et), 
// Gestational age in WKs based on LMP (m1_802d_et), 
// Interviewer calculate the GA in trimester (1st, 2nd or 3rd ) (m1_804), 
// Gestational age  based on LMP (Calc) (m1_ga), 
//  m1_trimester GA based on LMP from maternal care (mcard_ga_lmp) and 
//  LMP from maternal care (mcard_lmp)
 bro respondentid m1_date m1_801 m1_802a m1_802b_et m1_802c_et m1_802d_et m1_804 m1_ga m1_trimester mcard_ga_lmp mcard_lmp

*------------------------------------------------------------------------------	  
"Women health litreacy, HRQoL, Self reported health, family support"
        "Risky behavour and IPV"
*------------------------------------------------------------------------------
*Women health litery: based on the six health litracy queion**
tab m1_health_literacy // computed by cathrine and the list of six questions will be identified latter
		 
*Women health related quality of life during pregnancy(m1_205a to m1_205e)***	  
 // Assessed using the ERU-5Q-3L 
 // First, change each dimension of the HRQoL into a specific utility score, 
 // Considering the Zimbabwe utility tariff score for level 2 and level 3 responses, respectively
     *recode m1_205a, gen(anc_mobility) (0.056, 0.204) 
     *recode m1_205b, gen(pnc_selfcare) (0.092, 0.231)
     *recode m1_205c, gen (pnc_usual) (0.043, 0.135)
     *recode m1_205d, gen (pnc_pain)   (0.067, 0.302)
     *recode m1_205e, gen(pnc_anxiety) (0.046, 0.173)
	 
*EQ-5D-3L scoring using Zimbabwe Tariff 
* consider the analysis not incuding missing
*First generte Tthe complted HRQoL(complete_hrqol) if there is a missing data in each measurement including the skip option
gen complete_anc_hrqol = !missing(m1_205a, m1_205b, m1_205c, m1_205d, m1_205e) 
misstable summarize m1_205a m1_205b m1_205c m1_205d m1_205e // checking the missing data cleaned
   *then generate the each demesions score acoring

gen anc_mobility=cond(m1_205a==2,0.056, cond(m1_205a==3,0.204,0)) if complete_anc_hrqol
lab var anc_mobility "Women self-rated HRQoL mobility"
ta anc_mobility

gen anc_selfcare=cond(m1_205b==2,0.092, cond(m1_205b==3,0.231,0)) if complete_anc_hrqol
lab var anc_selfcare "Women self-rated HRQoL selfcare"
ta anc_selfcare

gen anc_usual=cond(m1_205c==2,0.043, cond(m1_205c==3,0.135,0)) if complete_anc_hrqol
lab var anc_usual "Women self-rated HRQoL usual activity"
ta anc_usual

gen anc_pain=cond(m1_205d==2,0.067, cond(m1_205d==3,0.302,0)) if complete_anc_hrqol
lab var anc_pain "Women self-rated HRQoL physical pain or discomfort"
ta anc_pain

gen anc_anxiety=cond(m1_205e==2,0.046, cond(m1_205e==3,0.173,0)) if complete_anc_hrqol
lab var anc_anxiety "Women self-rated HRQoL anxiety or depression"
ta anc_anxiety

*Considering the EQ-5D-3L Zebabawi tarrif utility score for HRQOL
gen anc_hrqol_score = 1 - anc_mobility - anc_selfcare - anc_usual - anc_pain - anc_anxiety ///
- cond(anc_mobility==0.204 | anc_selfcare==0.231 | anc_usual==0.135 | anc_pain==0.302 | anc_anxiety==0.173, 0.05, 0)
lab var anc_hrqol_score "EQ-5D-3L utility score of pregnant women"
replace anc_hrqol_score=round(anc_hrqol_score, 0.01)
ta anc_hrqol_score
summ anc_hrqol_score

* Create EQ-5D HRQoL categories based on the the untility score(eq5d_utility_score)
/**************************************************************************
 Reference for categorization:
- Studies in SSA (e.g., Jelsma et al. 2003; Abebe et al. 2020, Ethiopia)
  often classify HRQoL as:
       High: ≥0.85
       Moderate: 0.70–0.84
       Low: <0.70
**************************************************************************/
*Women HRQoL during pregnancy using three catagory: high, moderate and low
recode anc_hrqol_score 0/0.69=3 0.70/0.84=2 0.85/max=1, gen(anc_hrqol_cat)
lab var anc_hrqol_cat "HRQoL catagory of pregnant women"
lab def anc_hrqol_cat 1 "High ≥0.85" 2 "Moderate 0.70–0.84" 3 "Low <0.70"
lab val anc_hrqol_cat anc_hrqol_cat
tab anc_hrqol_cat
tab anc_hrqol_cat, summarize(anc_hrqol_score)

*By two catagorty: High and moderate/low
recode anc_hrqol_score 0/0.84=0 0.85/max=1, gen(anc_hrqol_2cat)
lab def anc_hrqol_2cat 1 "High ≥0.85" 0 "Moderate/low <0.84"
lab val anc_hrqol_2cat anc_hrqol_2cat
tab anc_hrqol_2cat

order complete_anc_hrqol anc_mobility anc_selfcare anc_usual anc_pain anc_anxiety ///
 anc_hrqol_score anc_hrqol_cat anc_hrqol_2cat, before (healthstatuss1_2cat)
 
****************************************************************************** 
*Summary stastics of HRQOl
summ anc_hrqol_score
summarize anc_hrqol_score, detail // for checking the value of skewness and kurtosis and mean and SD of HRQOL 
sktest anc_hrqol_score  //for checking the skwniss and kurtosis and Joint test of normaly distributed data 
                        /// Tthe (p-value of these test >0.05 indicates the HRQOL normaly distributed)

hist anc_hrqol_score
hist anc_hrqol_score, freq normal
histogram anc_hrqol_score, width(0.05) freq title("Distribution of EQ-5D Index")
mean anc_hrqol_score, over(m1_trimester) // by pregnancy status
mean anc_hrqol_score, over(education)

* Histogram with HRQoL cutpoints
hist anc_hrqol_score, freq normal addplot(line 0.70 0.85, lpattern(dash)) ///
    title("Distribution of EQ-5D-3L Index") ///
    xtitle("EQ-5D-3L Index") ytitle("Frequency")
	
hist anc_hrqol_score, freq normal ///
xline(0.70 0.85, lpattern(dash) lcolor(red)) title("Distribution of EQ-5D-3L Index (Zimbabwe Tariff)") ///
xtitle("EQ-5D-3L Index") ytitle("Frequency")

graph bar (count), over(anc_hrqol_cat) title("HRQoL Categories") ///
ytitle("Number of women") bar(1, color(teal)) bar(2, color(orange)) bar(3, color(emerald)) blabel(bar)

graph bar (percent), over(anc_hrqol_cat) title("Distribution of HRQoL Categories") ytitle("Percentage") blabel(bar) 

graph bar (percent), over(anc_hrqol_cat, gap(25) label(labsize(medium))) ///
title("Distribution of HRQoL Categories (EQ-5D-3L, Zimbabwe Tariff)") ///
ytitle("Percentage", size(medlarge)) ///
blabel(bar, format(%4.2f) size(medium) color(black)) ///
bar(1, color(ltblue)) bar(2, color(orange)) bar(3, color(gs12)) ///
legend(off) bargap(25) ylabel(, format(%4.0f))

*Save the graph as PNG and PDF
graph export "HRQoL_Distribution_EQ5D3L_Zimbabwe.png", width(2000) replace
graph export "HRQoL_Distribution_EQ5D3L_Zimbabwe.pdf", replace

*Mean HRQoLscore by subgroup (e.g., education or trimeter, pnc_depression)
 
 mean anc_hrqol_score, over(m1_trimester)
**********************************************************************************
*Optional Women HRQoL, using EQ-5D-3L
{gen anc_hrqol_score = 1 - cond(m1_205a==2,0.056, cond(m1_205a==3,0.204,0))///
 - cond(m1_205b==2,0.092, cond(m1_205b==3,0.231,0))///
 - cond(m1_205c==2,0.043, cond(m1_205c==3,0.135,0))///
 - cond(m1_205d==2,0.067, cond(m1_205d==3,0.302,0))///
 - cond(m1_205e==2,0.046, cond(m1_205e==3,0.173,0))///
 - cond(m1_205a==3 | m1_205b==3 | m1_205c==3 | m1_205d==3 | m1_205e==3, 0.05, 0)}
******************************************************************************** 	
*------------------------------------------------------------------------------
"Self-reported overall health status of pregant women (m1_201)"
*------------------------------------------------------------------------------
 gen healthstatuss1_2cat=m1_201
 recode healthstatuss1_2cat (1/2=1) (3/5=0)
 lab var healthstatuss1_2cat "201. Self-reported overall health status" 
 lab def healthstatuss1_2cat 1 "Excellent/very good" 0 "Good/fair/poor"
 lab val healthstatuss1_2cat healthstatuss1_2cat
 ta healthstatuss1_2cat

*Women's overall self-reported health stats across survey points times*
 gen healthstatus1= m1_201
 recode healthstatus1 1/2=1 3/5=0
 lab def healthstatus1 1 "Excellent/very good" 0 "Good/fair/poor"
 lab var healthstatus1 "201 Self-rated overall health status during pregnancy"
 lab val healthstatus1 healthstatus1
 tab healthstatus1
 
gen healthstatus2= m2_201_r1
recode healthstatus2 1/2=1 3/5=0
lab def healthstatus2 1 "Excellent/very good" 0 "Good/fair/poor"
lab var healthstatus2 "201 Self-rated overall health status during pregnancy"
lab val healthstatus2 healthstatus2
tab healthstatus2

gen healthstatus3= m2_201_r2
recode healthstatus3 1/2=1 3/5=0
lab def healthstatus3 1 "Excellent/very good" 0 "Good/fair/poor"
lab var healthstatus3 "201 Self-rated overall health status during pregnancy"
lab val healthstatus3 healthstatus3
tab healthstatus3

gen healthstatus4= m2_201_r3
recode healthstatus4 1/2=1 3/5=0
lab def healthstatus4 1 "Excellent/very good" 0 "Good/fair/poor"
lab var healthstatus4 "201 Self-rated overall health status during pregnancy"
lab val healthstatus4 healthstatus4
tab healthstatus4

gen healthstatus5= m2_201_r4
recode healthstatus5 1/2=1 3/5=0
lab def healthstatus5 1 "Excellent/very good" 0 "Good/fair/poor"
lab var healthstatus5 "201 Self-rated overall health status during pregnancy"
lab val healthstatus5 healthstatus5
tab healthstatus5

recode m2_201_r5 1/2=1 3/5=0, gen(healthstatus6) 
lab def healthstatus6 1 "Excellent/very good" 0 "Good/fair/poor"
lab var healthstatus6 "201 Self-rated overall health status during pregnancy"
lab val healthstatus6 healthstatus6
tab healthstatus6

recode m2_201_r6 1/2=1 3/5=0, gen(healthstatus7) 
lab def healthstatus7 1 "Excellent/very good" 0 "Good/fair/poor"
lab var healthstatus7 "201 Self-rated overall health status during pregnancy"
lab val healthstatus7 healthstatus7
tab healthstatus7

recode m2_201_r7 1/2=1 3/5=0, gen(healthstatus8) 
lab def healthstatus8 1 "Excellent/very good" 0 "Good/fair/poor"
lab var healthstatus8 "201 Self-rated overall health status during pregnancy"
lab val healthstatus8 healthstatus8
tab healthstatus8

recode m2_201_r8 1/2=1 3/5=0, gen(healthstatus9) 
lab def healthstatus9 1 "Excellent/very good" 0 "Good/fair/poor"
lab var healthstatus9 "201 Self-rated overall health status during pregnancy"
lab val healthstatus9 healthstatus9
tab healthstatus9

order healthstatus1 healthstatus2 healthstatus3 healthstatus4 healthstatus5 ///
healthstatus6 healthstatus7 healthstatus8 healthstatus9, after(healthstatuss1_2cat)

**Number of family with the mother that supports/help during this pregnancy**
 gen m1_familysupport=m1_508
 recode m1_familysupport (0/1= 1) (2/max=2)
 lab var m1_familysupport "508. family member who support you for help"
 lab def m1_familysupport 1 "Less than two" 2 "Two and more"
 lab val m1_familysupport m1_familysupport
 tab m1_familysupport
	
**Experience violence during pregnancy**
  // Asked for all mothers interviwed from q1101 to q1105
  // Physical violence (m1_1101:q1101)
     //160108/row-224 missed data
     *Type of perpertrator (q1102), asked if m1_1101 = 1
		
*Physical violence
 gen m1_physical_violence = m1_1101
 recode m1_physical_violence (1 = 1) (0 = 0)  
 lab var m1_physical_violence "Experience of physical violence in the current pregnancy"
 lab def m1_physical_violence 1 "Yes" 0 "No" 
 lab val m1_physical_violence m1_physical_violence  
 tab m1_physical_violence  //mother with id 10318/r-523" corrected as "no" instead of miseed.
		 
*Physical violene perpetrator (1 Partner, 2 family member 3 others)
 gen m1_perpeterater_Phy=.
 replace m1_perpeterater_Phy=1 if m1_1102_a==1 | m1_1102_e==1
 replace m1_perpeterater_Phy=2 if m1_1102_b==1 | m1_1102_c==1 | m1_1102_d==1
 replace m1_perpeterater_Phy=3 if m1_1102_f==1 | m1_1102_g==1 | m1_1102_h==1 |///
 m1_1102_i==1 |m1_1102_j==1 | m1_1102_96==1
 lab var m1_perpeterater_Phy "Perpeterater of physical violence"
 lab def m1_perpeterater_Phy 1 "Partener" 2 "Family" 3 "Others"
 lab val m1_perpeterater_Phy m1_perpeterater_Phy
 tab m1_perpeterater_Phy
		  	 
*Emotional violence (m1_1103/ m1_emotional_violence) 
   // 301636/row-571 missed data
   // m1_1103: q1103, 
   // Type of perpertrator (q1104), asked if m1_1103==1 
     *from m1_1104_a to  m1_1104_j:Strenger m1_1102_96 m1_1102_other: others // 
  // m1_1105: q1105: did HCWs disscus with you where you can seek support for these things? 
      bro repondentid  m1_1101 m1_1102 m1_1103 m1_1104 m1_1105   
 
*Emotional violence (m1_1103/ m1_emotional_violence)
 gen m1_emotional_violence = m1_1103
 recode m1_emotional_violence (1 = 1) (0 = 0)
 lab var m1_emotional_violence "Experience of emotional violence in the current pregnancy"
 lab def m1_emotional_violence 1 "Yes" 0 "No" 
 lab val m1_emotional_violence m1_emotional_violence
 tab m1_emotional_violence  
 //mother with id 10318/r-523" corrected as "no" instead of miseed 
   bro id m1_physical_violence m1_emotional_violence  m1_perpeterater_emo m1_perpeterater_Phy m1_violence 
		  
*Emotional violene perpetrator(m1_perpeterater_emo) 
 gen m1_perpeterater_emo=.
 replace m1_perpeterater_emo=1 if m1_1104_a==1 | m1_1104_e==1
 replace m1_perpeterater_emo=2 if m1_1104_b==1 | m1_1104_c==1 | m1_1104_d==1
 replace m1_perpeterater_emo=3 if m1_1104_f==1 | m1_1104_g==1 | m1_1104_h==1 |///
 m1_1104_i==1 | m1_1104_j==1 | m1_1104_96==1
 lab var m1_perpeterater_emo "Perpeterater of emotional violence"
 lab def m1_perpeterater_emo 1 "Partener" 2 "Family" 3 "Others"
 lab val m1_perpeterater_emo m1_perpeterater_emo
 tab m1_perpeterater_emo 
		  
*Physical or Emotional violence during Pregnancy
 egen m1_ipv_px_cat=rowtotal(m1_emotional_violence m1_physical_violence), missing 
 gen m1_ipv_px_cat=m1_violence_px
 recode m1_ipv_px_cat (1/2=1) (0=0)
 lab var m1_ipv_px_cat "Pregnant women expereienced IPV"
 lab def m1_ipv_px_cat 1 "Yes" 0 "No"
 lab val m1_ipv_px_cat m1_ipv_px_cat
 tab m1_ipv_px_cat 
 
 rename m1_violence_px_cat m1_ipv_px_cat
		
*Type of perpetrater during pregnancy (m1_violence_perpeterater) (1 Partner, 2 family member 3 others))
 gen m1_violence_perpeterater=.
 replace m1_violence_perpeterater=1 if m1_perpeterater_Phy==1 | m1_perpeterater_emo==1
 replace m1_violence_perpeterater=2 if m1_perpeterater_Phy==2 | m1_perpeterater_emo==2
 replace m1_violence_perpeterater=3 if m1_perpeterater_Phy==3 | m1_perpeterater_emo==3
 lab var m1_violence_perpeterater "Type of perpeterater for violence"
 lab def m1_violence_perpeterater 1 "Partner" 2 "Family member" 3 "Others"
 lab val m1_violence_perpeterater m1_violence_perpeterater
 tab m1_violence_perpeterater 
			
 bro id m1_enrollage marriedp  m1_physical_violence m1_perpeterater_Phy ///
 m1_emotional_violence m1_perpeterater_emo   m1_1101 m1_1103 m1_violence_px_cat ///
 m1_violence_perpeterater m2_202_r2 m2_202_r3
	
**Women with risky health behavour, including cigarettes, kaht and alcohol****
*Survey1
 gen m1_cigarette_r0=m1_901
 lab var m1_cigarette_r0  "901. Cigarette smoking"
 tab m1_cigarette_r0
	   
 gen m1_khat_r0=m1_903
 lab var m1_khat_r0  "903. Khat chewing"
 tab m1_khat_r0
	   
 gen m1_alcohol_r0=m1_905
 lab var m1_alcohol_r0  "905. Alcohol  used"
 tab m1_alcohol_r0

 egen subuses1=rowtotal(m1_cigarette_r0 m1_khat_r0 m1_alcohol_r0), mi 
 lab var subuses1 "Use of alcohol, khat or cigarette in current pregnancy" 
 tab subuses1
	  *180801/r-48, 301607/r-67, 201712/r-277, 71507/r-817 corrected as no insted of missed,
 gen subuse_cat1=subuses1  
 recode subuse_cat1 (0=0) (1/max=1)
 lab var subuse_cat1 "Pregnant women used any of alcohol, khat or cigarette "
 lab def subuse_cat1 1 " Yes" 0 "No"
 lab val subuse_cat1 subuse_cat1
 tab subuse_cat1 
				
*Survey2
gen m2_cigarette_r1=m2_206_r1
recode m2_cigarette_r1 (0/2=1) (3=0)
lab var m2_cigarette_r1 "206. Cigarette smoking"
	
gen m2_khat_r1=m2_207_r1
recode m2_khat_r1 (1/2=1) (3=0)
lab var m2_khat_r1 "207. Khat chewing"
	
gen m2_alcohol_r1=m2_208_r1
recode m2_alcohol_r1 (1/2=1) (3=0)
lab var m2_alcohol_r1 "208. Alcohol used"
  
egen subuses2=rowtotal(m2_cigarette_r1 m2_khat_r1 m2_alcohol_r1), missing
lab var subuses0"Use of alcohol, khat or cigarette in current pregnancy"
ta subuses2
 
gen subuse_cat2=subuses2
recode subuse_cat2 (0=0) (1/max=1)
lab var subuse_cat2 "Pregnant women used any of alcohol, khat or cigarette"
lab def subuse_cat2 1 " Yes" 0 "No"
lab val subuse_cat2 subuse_cat2
tab subuse_cat2
rename subuses2_cat subuse_cat2

*Survey3
 gen m2_cigarette_r2=m2_206_r2
 recode m2_cigarette_r2 (0/2=1) (3=0)
 lab var m2_cigarette_r2 "206. Cigarette smoking"
 
 gen m2_khat_r2=m2_207_r2
 recode m2_khat_r2 (1/2=1) (3=0)
 lab var m2_khat_r2 "207 Khat chewing"
 
 gen m2_alcohol_r2=m2_208_r2
 recode m2_alcohol_r2 (1/2=1) (3=0)
 lab var m2_alcohol_r2 "208. Alcohol used"
	
 egen subuses3=rowtotal(m2_cigarette_r2 m2_khat_r2 m2_alcohol_r2), mi
 lab var subuses3 "Use of alcohol, khat or cigarette in current pregnancy"
 
 gen subuse_cat3=subuses3
 recode subuse_cat3 (0=0) (1/max=1)
 lab var subuse_cat3 "Pregnant women used any of alcohol, khat or cigarette"
 lab def subuse_cat3 1 " Yes" 0 "No"
 lab val subuse_cat3 subuse_cat3
 tab subuse_cat3 
		
*Survey4
 gen m2_cigarette_r3=m2_206_r3
 recode m2_cigarette_r3 (0/2=1) (3=0)
 lab var m2_cigarette_r3 "206. Cigarette smoking"
 
 gen m2_khat_r3=m2_207_r3
 recode m2_khat_r3 (1/2=1) (3=0)
 lab var m2_khat_r3 "207. Khat chewing"
 
 gen m2_alcohol_r3=m2_208_r3
 recode m2_alcohol_r3 (1/2=1) (3=0)
 lab var m2_alcohol_r3 "208. Alcohol used"

 egen subuses4=rowtotal(m2_cigarette_r3 m2_khat_r3 m2_alcohol_r3), missing
 lab var subuses4 "Use of alcohol, khat or cigarette in current pregnancy"
 
gen subuse_cat4=subuses4
recode subuse_cat4 (0=0) (1/max=1)
lab var subuse_cat4 "Pregnant women used any of alcohol, khat or cigarette"
lab def subuse_cat4 1 " Yes" 0 "No" 
lab val subuse_cat4 subuse_cat4
tab subuse_cat4
	
*Survey5
gen m2_206_cigarette_r4=m2_206_r4
recode m2_cigarette_r4 (0/2=1) (3=0)
lab var m2_cigarette_r4 "206. Cigarette smoking"

gen m2_khat_r4=m2_207_r4
recode m2_khat_r4 (1/2=1) (3=0)
lab var m2_khat_r4 "207. Khat chewing"
 
rename m2_207_khat_r4 m2_khat_r4
gen m2_alcohol_r4=m2_208_r4
recode m2_alcohol_r4 (1/2=1) (3=0)
lab var m2_alcohol_r4 "208. Alcohol used"

egen subuses5=rowtotal(m2_cigarette_r4 m2_khat_r4 m2_alcohol_r4), missing
lab var subuses5 "Use of alcohol, khat or cigarette in current pregnancy"
 
gen subuse_cat5=subuses5
recode subuse_cat5 (0=0) (1/max=1)
lab var subuse_cat5 "Used any substance during the current pregnancy" 
lab def subuse_cat5 1 " Yes" 0 "No"
lab val subuse_cat5 subuse_cat5
tab subuse_cat5 

*Experienced of persistent substance used disorder across four survey points
egen pers_subuse_s1s4=rowtotal(subuse_cat1 subuse_cat2 subuse_cat3 subuse_cat4), mi
lab var pers_subuse_s1s4 "Persistently use substance across four survys"
tab pers_subuse_s1s4 
		
*Experienced of persistent substance used disorder across five survey points
 egen pers_subuse_s1s5=rowtotal(subuse_cat1 subuse_cat2 subuse_cat3 subuse_cat4 subuse_cat5), mi
 lab var pers_subuse_s1s5 "Persistently use substance across five survys"
 tab pers_subuse_s1s5 
		 	
*Survey6 
gen m2_206_cigarette_r5=m2_206_r5
recode m2_206_cigarette_r5 (0/2=1) (3=0)
lab var m2_206_cigarette_r5 "Cigarette smoke"

gen m2_207_khat_r5=m2_207_r5
recode m2_207_khat_r5 (1/2=1) (3=0)
lab var m2_207_khat_r5 "Khat chewing "
gen m2_208_alcohol_r5=m2_208_r5
recode m2_208_alcohol_r5 (1/2=1) (3=0)
lab var m2_208_alcohol_r5 "Alcohol used"

egen subuses6=rowtotal(m2_206_cigarette_r5 m2_207_khat_r5 m2_208_alcohol_r5), mi
lab var subuses6 "Use of alcohol, khat or cigarette in current pregnancy"

gen subuses6=subuses6
recode subuse_cat6 (0=0) (1/max=1)
lab var subuse_cat6 "Used any substance during the current pregnancy" 
lab val subuse_cat6 subuse_cat6  
tab subuse_cat6
	
*Survey7
gen m2_206_cigarette_r6=m2_206_r6
recode m2_206_cigarette_r6 (0/2=1) (3=0)
lab var m2_206_cigarette_r6 "Cigarette smoke"

gen m2_207_khat_r6=m2_207_r6
recode m2_207_khat_r6 (1/2=1) (3=0)
lab var m2_207_khat_r6 "Khat chewing "
		
gen m2_208_alcohol_r6=m2_208_r6
recode m2_208_alcohol_r6 (1/2=1) (3=0)
lab var m2_208_alcohol_r6 "Alcohol used"

egen subuses7=rowtotal(m2_206_cigarette_r6 m2_207_khat_r6 m2_208_alcohol_r6), mi
lab var subuses7 "Use of alcohol, khat or cigarette in current pregnancy"
      
gen subuse_cat7=subuses7
recode subuse_cat7 (0=0) (1/max=1)
lab var subuse_cat7 "Used any substance during the current pregnancy" 
tab subuse_cat7
		
*Survey8
gen m2_206_cigarette_r7=m2_206_r7
recode m2_206_cigarette_r7 (0/2=1) (3=0)
lab var m2_206_cigarette_r7 "Cigarette smoke"

gen m2_207_khat_r7=m2_207_r7
recode m2_207_khat_r7 (1/2=1) (3=0)
lab var m2_207_khat_r7 "Khat chewing "

gen m2_208_alcohol_r7=m2_208_r7
recode m2_208_alcohol_r7 (1/2=1) (3=0)
lab var m2_208_alcohol_r7 "Alcohol used"
		
egen subuses8=rowtotal(m2_206_cigarette_r7 m2_207_khat_r7 m2_208_alcohol_r7), mi
lab var subuses8 "Use of alcohol, khat or cigarette in current pregnancy"

gen subuse_cat8=subuses8
recode subuse_cat8 (0=0) (1/max=1)
lab var subuse_cat8 "Used any substance during the current pregnancy" 
tab subuse_cat8

**Received iron folic at least for xx months during pregnancy** 
 // First make the variable catagory in module one (m1_713a) recoded as 
 // Yes and No from three catagories, then generate new variable similar with 
 // varables in module two (m2_603_r1 to else)
	
*IFA provided at module1
recode m1_713a (1/2=1) (3=0)  
lab def m1_713a  1 "Yes" 0 "No"  // 1 is directly given or prescribed 
lab val m1_713a m1_713a
tab m1_713a  // mother with id 71534/r-412 corrected as "no" instead of missed
		  
gen irontakens1=m1_713a 
lab var irontakens1 "603. Pregnant women took IFA pills" 
lab def irontakens1 1 "Yes" 0 "No"
lab val irontakens1 irontakens1
tab irontakens1
		  
rename m2_603_r1 irontakens2 //603.currently taking iron & folic acid pills
lab var irontakens2 "603. Pregnant women took IFA pills" 

rename m2_603_r2 irontakens3 // 603. currently taking iron & folic acid pills
lab var irontakens3 "603. Pregnant women took IFA pills" 	 
rename m2_603_r3 irontakens4 //603. currently taking iron & folic acid pills
lab var irontakens4 "603. Pregnant women took IFA pills" 		 
rename m2_603_r4 irontakens5 //603. currently taking iron & folic acid pills
lab var irontakens5 "603. Pregnant women took IFA pills"
rename m2_603_r5 irontakens6 //603. currently taking iron & folic acid pills
lab var irontakens6 "603. Pregnant women took IFA pills"
rename m2_603_r6 irontakens7 //603. currently taking iron & folic acid pills
lab var irontakens7 "603. Pregnant women took IFA pills"
rename m2_603_r7 irontakens8 //603. currently taking iron & folic acid pills
lab var irontakens8 "603. Pregnant women took IFA pills"

order irontakens6 irontakens7 irontakens8, after(irontakens5)
***considering these variables as repeated measure of iron intake and 
 //changed into long form as one varable  as "irontaken;

 tab irontakens2 	
******************************************************************************

**Chronic and other health problems of pregnant women***

**Women's history of chronic illness (m1_202a to m1_203_et, m1_203_other)**
 bro id m1_202a m1_202b m1_202c m1_202d m1_202e m1_202f_et m1_202g_et m1_203_et m1_203_other 
  //301623/r-53 for m1_202a, 160135/r-555 for m1_202g_et, 
  //231636/r-699 for m1_202e, 121506/ r-701 for m1_202f_et and 
  //m1_202g_et, 141006/r-917 for m1_202f_et were missed. 
  //However, i corrected these data as "No" 
  
 //The following diseases, which are specified by other response (m1_203_other)
 //were not considered for analysis as chronic illness:
   *Gastritis, Anemia, Chgara, chigat, Chronic Sinusitis and tonsil, Dibite, Gastritis, and Gastrointestinal tract  
	 
**pregnant women who had a history of at least one chronic illness*
     //including any of DM, HPN, cardiac, mental health, HIV, Hepatitis B, Renal health, and other chronic illnesses		     
 egen m1_Hx_chronic_illness=rowtotal (m1_202a m1_202b m1_202c m1_202d m1_202e m1_202f_et m1_202g_et m1_203_et), missing // including other chronic illness
 recode m1_Hx_chronic_illness (0/0=0) (1/max=1)
 lab var m1_Hx_chronic_illness "Women's history of chronic illness"
 lab def m1_Hx_chronic_illness 1 "Yes, at least one" 0 "No illness"
 lab val m1_Hx_chronic_illness m1_Hx_chronic_illness
 tab m1_Hx_chronic_illness
	    
**Newly diagnosed with chronic illness in the current pregnancy, such as; 
  //DM, syphilis, HIV, Asthma, elevated PB
   
*Newly diagnosed syphilis, both from maternal card and self-report of the mother
gen m1_syphilis=m1_710b
recode m1_syphilis (1=1) (2=0)
lab var m1_syphilis " 710b. newly diagnosed syphilis +ve at 1st ANC " // including the maternal card 
tab m1_syphilis

*Newly diagnosed HIV at ANC from self-report of the mother and card review
gen m1_hiv=m1_708b
recode m1_hiv (1=1) (2=0)
lab var m1_hiv " 708b. newly diagnosed HIV positive at 1st ANC"
tab m1_hiv

*Newly diagnosed DM at ANC from self-report of the mother/mcard
gen m1_dm=m1_711b
recode m1_dm (1=1) (2=0)
lab var m1_dm " 711b. newly diagnosed DM at 1st ANC"
tab m1_dm
	  
*Experience asthma in the current pregnancy from the mother self self-report
gen m1_asthema =m1_8e_et
lab var m1_asthema " Experienced asthma in current pregnancy"
tab m1_asthema
      
*Elevated gestational HPN (m1_gestational_hpn)
  // measured by data collectors 3 times with five five-minute time intervals on the same day
  
  // Systolic blood pressure (m1_bp_time_1_systolic m1_bp_time_2_systolic m1_bp_time_3_systolic) coded as  (m1_sbp)
      *Has 6 missied data in the 2nd (71513/row-239,71512/row-460, 100707/row-515, 100735/row-516, 100733/row-673, 100703/row-776) and 
      *7 missed data in the third measurement of SBP (71513/row-239,71512/row-460, 100707/row-515, 100735/row-516, 100733/row-673, 100703/row-776, 71507/row-816).
	  
*Diastolic (m1_bp_time_1_diastolic m1_bp_time_2_diastolic m1_bp_time_3_diastolic), coded as m1_dbp
  // Has 6 missed data for 2nd measurement (71513/r-239, 71512/r-460,100707/row-515, 100735/row-516, 100733/row-67, 100703/r-776 ), and 
  // Has 8 missed data for 3rd measurement (100730/r-159, 71513/r-239, 71512/r-460,100707/row-515, 100735/row-516, 100733/r-673,100703/r-776, 71507/r-816)
  // As a result, the mean value of 68 was imputed for those missing data
  // As a result, the mean value was taken for each measurement of SBP  
*After doing these average SBP and DBP were computed as follows;
*Average systolic blood pressure
gen m1_sbp=(m1_bp_time_1_systolic + m1_bp_time_2_systolic + m1_bp_time_3_systolic)/3 
lab var m1_sbp "The average systolic blood pressure"
tab m1_sbp
   
*Average diastolic blood pressure 
gen m1_dbp=(m1_bp_time_1_diastolic + m1_bp_time_2_diastolic + m1_bp_time_3_diastolic)/3
lab var m1_dbp "The average diastolic blood pressure"
tab m1_dbp
   
*Elevated systolic blood pressure  
gen m1_elevated_sbp=m1_sbp
recode m1_elevated_sbp (min/139=0) (140/max=1)
lab var m1_elevated_sbp "Elevated systolic blood pressure >=140 mmHg" 
lab def m1_elevated_sbp 1 "Yes" 0 "No"
lab val m1_elevated_sbp m1_elevated_sbp
tab m1_elevated_sbp
   
*Elevated dystorlic blood pressure
gen m1_elevated_dbp=m1_dbp
recode m1_elevated_dbp (min/89=0) (90/max=1)
lab var m1_elevated_dbp "Elevated diastolic blood pressure >=90 mmHg" 
lab def m1_elevated_dbp 1 "Yes" 0 "No"
lab val m1_elevated_dbp m1_elevated_dbp
tab m1_elevated_dbp
   
*Gestational hypertension
gen m1_gestational_hpn=.
replace m1_gestational_hpn=1 if m1_elevated_sbp==1 | m1_elevated_dbp==1
replace m1_gestational_hpn=0 if m1_elevated_sbp==0 & m1_elevated_dbp==0
lab var m1_gestational_hpn "Gestational HPN sbp >140 mmHg or dbp >89 mmHg" 
lab def m1_gestational_hpn 1 "Yes" 0 "No"
lab val m1_gestational_hpn m1_gestational_hpn
tab m1_gestational_hpn
	
*For checking the Gestational HPN by trimester level 
tab m1_gestational_hpn m1_trimester, row  // using row table
tab m1_gestational_hpn m1_trimester, column // column table	
	  
*Women with chronic illness diagnosed at ANC*
egen m1_cur_chronic_illness= rowtotal (m1_syphilis m1_hiv m1_dm m1_asthema m1_gestational_hpn), mi
recode m1_cur_chronic_illness (0/0=0) (1/max=1)
lab var m1_cur_chronic_illness "Women with any chronic illness diagnosed at ANC visit"
lab def m1_cur_chronic_illness 1 "Yes" 0 "No"
lab val m1_cur_chronic_illness m1_cur_chronic_illness
tab m1_cur_chronic_illness 
tab m1_asthema
 
**Overall chronic illnesses of women: including history and currently identified 
egen m1_chronic_illness=rowtotal (m1_202a m1_202b m1_202c m1_202d ///
m1_202e m1_202f_et m1_202g_et m1_203_et m1_syphilis m1_hiv m1_dm ///
m1_gestational_hpn m1_asthema), mi // including other chronic illness
recode m1_chronic_illness (0=0) (1/max=1)
lab var m1_chronic_illness "Women with any chronic illness previously plus currently" 
lab def m1_chronic_illness 1 "Yes, at least one" 0 "No"
lab val m1_chronic_illness m1_chronic_illness
tab m1_chronic_illness
   //	identifed by interview and looking at maternal card
*******************************************************************************
 
**Women with chronic medical problems experienced in current pregnancy*
  // such as (elevated BP, syphilis, DM, HIV)
{//Gestational DM from mother self-reported and maternal card review 
 gen m1_dm1=.
 replace m1_dm1=1 if m1_711b==1|m2_505e_r1==1|m2_505e_r2==1|m2_505e_r3==1|m2_505e_r4==1|m2_505e_r5==1|m2_505e_r6==1|m2_505e_r7==1
 lab var m1_dm1 " gestational DM including those who evalated upto round 8"
 drop m1_dm1
	
 gen m1_dm=.
 replace m1_dm=1 if m1_711b==1|mcard_dm==1|m2_505e_r1==1|m2_505e_r2==1|m2_505e_r3==1 
 lab var m1_dm " gestational DM incuding those mother followed upto m2 round3 "
 drop m1_dm

*Number of prgenant women with syphills both from maternal card and m2 data upto round 3
 gen m1_syphilis=.
 replace m1_syphilis=1 if m1_710b==1|m2_505d_r1==1|m2_505d_r2==1|m2_505d_r3==1 //3 mother with syphilis postive from mcard is added in m1_710b
 lab var m1_syphilis "Pregnant women with syphilis both from m1 m2and maternal card"
   drop m1_syphilis
 
*Number of pregant women with HIV
 gen m1_hiv=.
 replace m1_hiv=1 if m1_708b==1|m2_505b_r1==1|m2_505b_r2==1|m2_505b_r3==1
 lab var m1_hiv "Pregnant women with HIV postive both from m1 m2 and maternal card"
 drop m1_hiv}  

{ //Pregant women at least one chronic medical problems in the current px** droped
 gen m1_medical_problems=.
 replace m1_medical_problems=1 if m1_dm==1|m1_syphilis==1|m1_hiv==1|m1_gestational_hpn==1
 lab var m1_medical_problems " pregant women with medical problem of DM or HIV or syphilis or elevated BP"
   drop m1_medical_problems 
 
 gen m1_atleast_one_medical=m1_medical_problems // droped
 recode m1_atleast_one_medical (1=1) (.=0)
 lab def m1_atleast_one_medical 1 "Yes" 0 "No"
 lab var m1_atleast_one_medical " Pregnant women at least one medical problems"
 lab val m1_atleast_one_medical m1_atleast_one_medical
    drop m1_atleast_one_medical} // 
********************************************************************************   	

**Other medical problems during the current pregnancy**
 
*Maternal amemia* 
  // Considered from Hgb level in maternal cared (m1_1307) and, 
  // tested by data collector (m1_1309)*
  bro m1_1306 m1_1307 m1_1308 m1_1309
	 
*Anemia from maternal card
 gen m1_anemia_mcard=m1_1307
 recode m1_anemia_mcard (0/10.99=1) (11/max=0)      
 lab var m1_anemia_mcard "Anemia from the maternal card"
 lab def m1_anemia_mcard 1 "Yes" 0 "No"
 lab val m1_anemia_mcard m1_anemia_mcard
 tab m1_anemia_mcard

*Anemia tested by data collectors (if the mother had no tested for anemia during ANC in the facility)
 gen m1_anemia_dc_test=m1_1309
 recode m1_anemia_dc_test (0/10.99=1) (11/max=0)      
 lab var m1_anemia_dc_test "Anemia evaluted by data collector"
 lab def m1_anemia_dc_test 1 "Yes" 0 "No"
 lab val m1_anemia_dc_test m1_anemia_dc_test
 tab m1_anemia_dc_test
	 
*Anemia during prgancy (from maternal card or tested by DC) only baseline**
 gen m1_anemia_px=.
 replace m1_anemia_px=1 if m1_anemia_mcard==1 | m1_anemia_dc_test==1  
 replace m1_anemia_px=0 if m1_anemia_mcard==0 | m1_anemia_dc_test==0 
 lab var m1_anemia_px "Anemia during prgnancy"
 lab def m1_anemia_px 1 "Anemic" 0 "No anemic"
 lab val m1_anemia_px m1_anemia_px
 tab m1_anemia_px 
 tab m1_anemia_px_edit 
 tab m1_anemia_px_edit_01 
 tab m1_anemia_px_edit_02 //only use for regressoon
	  
*Overall anemia during prgancacy from m1 to m2_r3  
{ gen anemia_prgnancy_m1_3=.
  replace anemia_prgnancy_m1_3=1 if m1_anemia_px==1 | m2_505a_r1==1 | m2_505a_r2==1 | m2_505a_r3==1
  replace anemia_prgnancy_m1_3=0 if m1_anemia_px==0 & m2_505a_r1==0 & m2_505a_r2==0 & m2_505a_r3==0
  lab var anemia_prgnancy_m1_3 "Anemia during prgnancy"
  lab def anemia_prgnancy_m1_3 1 "Anemic" 0 "No anemic"
  lab val anemia_prgnancy_m1_3 anemia_prgnancy_m1_3
	  drop anemia_prgnancy_m1_3  }    // not used
		
*Nutritional status of pregant women using MUAC(<23cm)**
 tab m1_malnutrition   
 // mother with id 71507/r-816 corrected as "No" instead of missed,

*Womens Boddy mass index at enrolement*
 {gen m1_bmi_cat= m1_bmi
 recode m1_bmi_cat (min/18.499=1) (18.50/24.99=2) (25/29.99=3)(30/max=4)
 lab var m1_bmi_cat "Body mass index of the pregnant women"
 lab def m1_bmi_cat 1 "Underweight < 18.5" 2 "Normal 18.5-24.99" 3 "Over weight 25.0-29.99" 4 "Obesity >=30"
 lab val m1_bmi_cat m1_bmi_cat
 tab m1_bmi_cat} // not generated
	
*Women nutritional status using BMI**   
 gen m1_bmi_low=m1_bmi
 recode m1_bmi_low (min/18.499=1) (18.50/max=0)
 lab var m1_bmi_low "Body mass index below 18.5kg/m2 in the current pregnancy"
 lab def m1_bmi_low 1 "Yes" 0 "No"
 lab val m1_bmi_low m1_bmi_low
 tab m1_bmi_low
 bro m1_bmi_low m1_bmi
	  
*Women with obesity in the current pregnancy*
 gen m1_bmi_obesity=m1_bmi
 recode m1_bmi_obesity (30/max =1) (min/29.99=0)  //
 lab def m1_bmi_obesity 1 "Yes" 0 "No"
 lab var m1_bmi_obesity "Body mass index >=30kg/m2 in the current pregnancy"
 lab val m1_bmi_obesity m1_bmi_obesity
 tab m1_bmi_obesity  
	  
*Women undernutrition/underweight:consideringh MUAC <23cm  or BMI <18.5*
 egen m1_nutritional_status=rowtotal(m1_bmi_low m1_malnutrition), mi
 recode m1_nutritional_status 0=0 1/max=1, gen (m1_undernutrition)
 lab var m1_undernutrition "Women with undernutrition using  MUAC <23cm or BMI <18.5"
 lab def m1_undernutrition 1 "Yes" 0 "No"
 lab val m1_undernutrition m1_undernutrition
 tab m1_undernutrition  //mother with id 71507_816 corrected as "No" instead of missed.

	order m1_bmi_low m1_muac m1_malnutrition m1_undernutrition, before (m1_bmi_obesity)
********************************************************************************   
***Women Obstetrics history and current health conditions/problems***//   
**Women's obstetric history**
*The following cases will be treated as incompletely recorded due to inconsistent and incomplete records about the history of the women's obstetric history
		 71509/row-546, 251706/row-584
         211432/row-588, 241916/row-691
         160130/row-867, 190135/row-939
		 
*Women gestational age an enrollment(m1_trimester) *
 tab m1_trimester
	 
*Known multiple pregnancy**
 gen m1_twinpregnancy= m1_805
 recode m1_twinpregnancy (2/3=1) (1=0) 
 lab var m1_twinpregnancy "805. Women with multiple pregnancy"
 lab def m1_twinpregnancy 1 "Yes" 0 "No"
 lab val m1_twinpregnancy m1_twinpregnancy 
 tab m1_twinpregnancy
	   
*Women aged <20 years and >35 years of age in the current pregnancy
 gen m1_pxage_risk=m1_enrollage
 recode m1_pxage_risk (min/19 36/max =1) (20/35=0)  // this variable should be generated for original data
 lab def m1_pxage_risk 1 "<20 or >35 years" 0 "21-35 years"
 lab var m1_pxage_risk "Women age at this pregnancy"
 lab val m1_pxage_risk m1_pxage_risk
 tab m1_pxage_risk
	 
*Gravidity (m1_gravidity) *
  // An 18-year-old mother with ID 10318/row-523 was documented as primigravida. However, the mother documented as having alive babies at m1_1009. 
  //As a result, I corrected the data applied for the primigravida mother 
 gen m1_gravidity=m1_1001
 recode m1_gravidity (1=1) (2/9=2)
 lab var m1_gravidity "Number of pregnancy, including the current"
 lab def m1_gravidity 1 "Primi-gravida" 2 "Multi-gravida"
 lab val m1_gravidity m1_gravidity
 tab m1_gravidity

*Parity (m1_1002_parity)*
 gen m1_parity=m1_1002_cl
 recode m1_parity (0=.) (1=1) (2=2) (3/max=3)
 lab var m1_parity "1002 women with birth alive or died"
 tab m1_parity

*Number of children born alive (m1_1003_born_alive)*
 gen m1_bornalivebaby=m1_1003_cl
 recode m1_bornalivebaby (0=.) (1/2=1) (3/max=2)
 lab var m1_bornalivebaby "1003 number of babies born alive"
 lab def m1_bornalivebaby 1 "less than three babies" 2 "Three and more babies"
 lab val m1_bornalivebaby m1_bornalivebaby
 tab m1_bornalivebaby

*History of pregnancy loss after 20 weeks of gestational age (m1_1004_hx_pxloss)*
 gen m1_hx_pxloss=m1_1004_g
 recode m1_hx_pxloss (1=1) (0=0)
 lab var m1_hx_pxloss "1004 Hx of pregnancy loss > 20 weeks of gestation"
 lab def m1_hx_pxloss 1 "Yes" 0 "No"
 lab val m1_hx_pxloss m1_hx_pxloss
 tab m1_hx_pxloss

*History of abortion (m1_abortion_cl)*
 lab var m1_abortion_cl "History of abortion" // these mothers with 71509/r-546, 211432/r-588 160130/r-867 updated as *no abortion* 
 tab m1_abortion_cl

*For computing history of abortion, the following variables were considered as follows: 
 m1_gravidity==2 & m1_1002_cl==0 & m1_parity==. & m1_1003_cl==. & m1_1004==yes|m1_1004==no 
	 
*History of stillbirths (m1_stillbirths_cl)*
 lab var m1_stillbirths_cl "History of stillbirths"
 tab m1_stillbirths_cl
	   
*For computing the history of stillbirths, the following variables were considered: 
   m1_gravidity==2 & m1_1002_cl>1 & m1_1003_cl==0 & m1_bornalivebaby=. & m1_hx_pxloss==yes	
   
*History of preterm birth (m1_1005_hx_ptb)*
 gen m1_hx_preterm= m1_1005_g
 recode m1_hx_preterm (1=1) (0=0)
 lab var m1_hx_preterm "1005 history of preterm birth"
 lab def m1_hx_preterm 1 "Yes" 0 "No"
 lab val m1_hx_preterm m1_hx_preterm
 tab m1_hx_preterm 

*History of delivery complication (m1_1006_hx_deliverycomp)*
 gen m1_hx_deliverycomp=m1_1006_g
 recode m1_hx_deliverycomp (1=1) (0=0)
 lab var m1_hx_deliverycomp "1006 history of delivery complication"
 lab def m1_hx_deliverycomp 1 "Yes" 0 "No"
 lab val m1_hx_deliverycomp m1_hx_deliverycomp
 tab m1_hx_deliverycomp
	   
*History of CS delivery (m1_1007_hx_CSdelivery)*
 gen m1_hx_csdelivery=m1_1007_g
 recode m1_hx_csdelivery (1=1) (0=0)
 lab var m1_hx_csdelivery "1007 history of CS delivery"
 lab def m1_hx_csdelivery 1 "Yes" 0 "No"
 lab val m1_hx_csdelivery m1_hx_csdelivery
 tab  m1_hx_csdelivery

*History of prolonged labour (m1_1008_hx_prolongedlabor)*
 gen m1_hx_prolongedlabor=m1_1008_g
 recode m1_hx_prolongedlabor (1=1) (0=0)
 lab var m1_hx_prolongedlabor "1008 history of prolonged labour"
 lab def m1_hx_prolongedlabor 1 "Yes" 0 "No"
 lab val m1_hx_prolongedlabor m1_hx_prolongedlabor
 tab m1_hx_prolongedlabor

*Number of alive children (m1_1009_num_alivechild)*
 gen m1_num_alivechild= m1_1009_cl
 recode m1_num_alivechild (0=.) (1/2=1) (3/max=2)
 lab var m1_num_alivechild "1009 number of current alive child"
 lab def m1_num_alivechild 1 "Less than three children" 2 "Three and more children"
 lab val m1_num_alivechild m1_num_alivechild
 tab m1_num_alivechild
		   
*History of neonatal death (m1_1010_hx_neonataldeath)*
 gen m1_hx_neonataldeath=m1_1010_g
 recode m1_hx_neonataldeath (1=1) (0=0)
 lab var m1_hx_neonataldeath "1009 history of neonatal death"
 lab def m1_hx_neonataldeath 1 "Yes" 0 "No"
 lab val m1_hx_neonataldeath m1_hx_neonataldeath
 tab m1_hx_neonataldeath
	   
*History of congenital anomalies (m1_hx_congenitalanomal)*
 gen m1_hx_congenitalanomal= m1_1_10_et_g
 recode m1_hx_congenitalanomal (1=1) (0=0)
 lab var m1_hx_congenitalanomal "Eth-1-10 history of congenital anomalies"
 lab def m1_hx_congenitalanomal 1 "Yes" 0 "No"
 lab val m1_hx_congenitalanomal m1_hx_congenitalanomal
 tab m1_hx_congenitalanomal

*For checking the history of obstetric
   bro id m1_enrollage m1_gravidity m1_1002 m1_1003 m1_1004 stillbirths m1_1005 m1_1006 m1_1007 m1_1008 m1_1009 m1_1010 m1_1_10_et // unclean
   bro id m1_gravidity m1_parity m1_bornalivebaby m1_hx_pxloss m1_abortion_cl m1_stillbirths_cl m1_hx_preterm m1_hx_deliverycomp m1_hx_csdelivery m1_hx_prolongedlabor m1_1num_alivechild  m1_hx_neonataldeath m1_hx_congenitalanomal // corrected and cleaned data
 

//Pregnant women who had at least one poor obestetric history: considering unclean data and excluding history of abortion (m1_aborition_cl)*
 {egen m1_poor_obst_hxun=rowtotal(stillbirths m1_1005 m1_1006 m1_1007 m1_1008  m1_1010 m1_1_10_et), missing                   
 recode m1_poor_obst_hxun (0=0) (1/max=1)
 lab var m1_poor_obst_hxun "At least one poor obstetric history using unclean data without abortion"
 lab def m1_poor_obst_hxun 1 "Yes" 0 "No"
 lab val m1_poor_obst_hxun m1_poor_obst_hxun
    drop m1_poor_obst_hxun }  // not comuted/ generated
	  
*Pregnant women who had poor obstetric history: using clean variables and including history of abortion (m1_abortion_cl)	  
 egen m1_hx_poorobsteric=rowtotal(m1_abortion_cl m1_stillbirths_cl m1_hx_preterm m1_hx_deliverycomp m1_hx_csdelivery m1_hx_prolongedlabor m1_hx_neonataldeath m1_hx_congenitalanomal), missing 
 lab var m1_hx_poorobsteric "Women's obstetrics history inluding 8 obstetrics problems"
 tab m1_hx_poorobsteric
	 
*Women had at least one poor obestetrics problem
 gen m1_hx_poorobst_cat1=m1_hx_poorobsteric
 recode m1_hx_poorobst_cat1 (0=0) (1/max=1)
 lab var m1_hx_poorobst_cat1 "Experienced at least one obstetric problem including abortion" 
 lab def m1_hx_poorobst_cat1 1 "Yes" 0 "No"
 lab val m1_hx_poorobst_cat1 m1_hx_poorobst_cat1
 tab  m1_hx_poorobst_cat1
	
*women had al least two obestetrics problem
 gen m1_hx_poorobst_cat2=m1_hx_poorobsteric
 recode m1_hx_poorobst_cat2 (0/1=0) (2/max=1)
 lab var m1_hx_poorobst_cat2 "Experienced at least two obstetric problems including abortion" 
 lab def m1_hx_poorobst_cat2 1 "Yes" 0 "No"
 lab val m1_hx_poorobst_cat2 m1_hx_poorobst_cat2
 tab m1_hx_poorobst_cat2 // not used for pregnancy risk group
		
*Pregnant women who had at least one poor obstetric history: using clean variables without a history of abortion
 {egen m1_poor_obst_hx1=rowtotal(m1_stillbirths_cl m1_hx_preterm m1_hx_deliverycomp m1_hx_csdelivery m1_hx_prolongedlabor m1_hx_neonataldeath m1_hx_congenitalanomal), missing 
 recode m1_poor_obst_hx1(0=0) (1/max=1)
 lab var m1_poor_obst_hx1 "Experienced at least one obstetric problem excluding abortion"
 lab def m1_poor_obst_hx1 1 "Yes" 0 "No"
 lab val m1_poor_obst_hx1 m1_poor_obst_hx1
 tab m1_poor_obst_hx1 } //not generated

*Experiance of common health problem during prgnancy (m1_comhealth_problem:*including any of naussa, heart burn, back pain and leg cramp in current px 
 egen m1_comhealth_problem=rowtotal(m1_813a m1_813b m1_813c m1_813d), missing
 recode m1_comhealth_problem (0/0=0) (1/4=1)
 lab var m1_comhealth_problem "Experienced any common health problems during prgnancy"
 lab def m1_comhealth_problem 1 "Yes" 0 "No"
 lab val m1_comhealth_problem m1_comhealth_problem
 tab m1_comhealth_problem	  
	  
*Pregnanat women had risk health factors at enrollment*
 tab m1_pxage_risk
 tab m1_anemia_px
 tab m1_malnutrition
 tab m1_chronic_illness // including
 tab m1_twinpregnancy
 tab  tab m1_hx_poorobst_cat1 // including 

egen m1_risk_health_px=rowtotal (m1_pxage_risk m1_anemia_px m1_malnutrition ///
 m1_chronic_illness m1_twinpregnancy m1_hx_poorobst_cat1), mi
lab var m1_risk_health_px "Women with pregnancy risky factors)"
ta m1_risk_health_px 
 
recode m1_risk_health_pregnancy 0=1 1=2 2/4=3, gen (m1_risk_health_px_cat)
lab def m1_risk_health_px_cat  1 "No risk" 2 "One risk" 3 "Two or more risk"  
tab m1_risk_health_px_cat m1_risk_health_px_cat 
ta m1_risk_health_px_cat

**Women with obstetric problems during the current pregnancy at different survey points**
  //including experience of any of the following: 
     * Preeclampsia /m1_8a_et/m2_204a_et_r1, 
     * Hyperemesis gravidarum/m1_8b_et/m2_204c_et_r1, 
     * Oligo/polyhydramnios/m1_8d_et/m2_204f_et_r1, 
     * RH isoimmunization/m1_8f_et/ m2_204h_et_r1
	   
*Survey one
egen obstproblems1=rowtotal(m1_8a_et m1_8b_et m1_8d_et m1_8f_et), missing
lab var obstproblems1 "Experience of obstetric problems" 
tab obstproblems1 
rename s1_obs_problemsm1_cat obstproblems1
	
 gen obstproblems1_cat=obstproblems1
 recode obstproblems1_cat (0/0=0) (1/max=1)
 lab var obstproblems1_cat "Experienced at least one obstetric problem"
 lab def obstproblems1_cat 1 "Yes" 0 "No"
 lab val obstproblems1_cat obstproblems1_cat
 tab obstproblems1_cat 
 rename obstproblems1_cat obstproblem_cat1

*Survey two
egen obstproblems2=rowtotal(m2_204a_et_r1 m2_204c_et_r1 m2_204f_et_r1 m2_204h_et_r1), mi
lab var obstproblems2 "Experience of obstetric problems" 
tab obstproblems2
	
gen obstproblems2_cat=obstproblems2
recode obstproblems2_cat (0/0=0) (1/max=1)
lab var obstproblems2_cat "Experienced at least one obstetric problem"
lab def obstproblems2_cat 1 " Yes" 0 "No"
lab val obstproblems2_cat obstproblems2_cat

rename obstproblems2_cat obstproblem_cat2
 
*Survey three
egen obstproblems3=rowtotal(m2_204a_et_r2 m2_204c_et_r2 m2_204f_et_r2 m2_204h_et_r2), mi
lab var obstproblems3 "Experience of obstetric problems" 
tab obstproblems3
	
gen obstproblems3_cat=obstproblems3
recode obstproblems3_cat (0/0=0) (1/max=1)
lab var obstproblems3_cat "Experienced at least one obstetric problem"
lab def obstproblems3_cat 1 " Yes" 0 "No"
lab val obstproblems3_cat obstproblems3_cat
rename obstproblems3_cat obstproblem_cat3
 
*Survey four
egen obstproblems4=rowtotal(m2_204a_et_r3 m2_204c_et_r3 m2_204f_et_r3 m2_204h_et_r3), missing
lab var obstproblems4 "Experience of obstetric problems" 
tab obstproblems4
	
gen obstproblems4_cat=obstproblems4
recode obstproblems4_cat (0/0=0) (1/max=1)
lab var obstproblems4_cat "Experienced at least one obstetric problem"
lab def obstproblems4_cat 1 " Yes" 0 "No"
lab val obstproblems4_cat obstproblems4_cat
tab obstproblems4_cat 
rename obstproblems4_cat obstproblem_cat4
	
*Survey five
egen obstproblems5=rowtotal(m2_204a_et_r4 m2_204c_et_r4 m2_204f_et_r4 m2_204h_et_r4), mi
lab var obstproblems5 "Experience of obstetric problems"
tab obstproblems5
	
gen obstproblems5_cat=obstproblems5
recode obstproblems5_cat (0/0=0) (1/max=1)
lab var obstproblems5_cat "Experienced at least one obstetric problem"
lab def obstproblems5_cat 1 " Yes" 0 "No"
lab val obstproblems5_cat obstproblems5_cat
tab  obstproblems5_cat
rename obstproblems5_cat obstproblem_cat5

*Experience of persistent obstetric problems across four survey times
egen pers_obstpro_s1s4=rowtotal(obstproblem_cat1 obstproblem_cat2 obstproblem_cat3 obstproblem_cat4), mi
lab var pers_obstpro_s1s4 "Persistent experience of obstetric problem across 4 surveys"
tab pers_obstpro_s1s4    

*Experience of persistent obstetric problems across five survey times
egen pers_obstpro_s1s5=rowtotal(obstproblem_cat1 obstproblem_cat2 obstproblem_cat3 obstproblem_cat4 obstproblems5_cat), mi
lab var pers_obstpro_s1s5 "Persistent experience of obstetric problem across 5 surveys"
tab pers_obstpro_s1s5  
	
*//Survey six
egen obstproblems6=rowtotal(m2_204a_et_r5 m2_204c_et_r5 m2_204f_et_r5 m2_204h_et_r5), mi
lab var obstproblems "Experience of obstetric problems" 

recode obstproblem_cat6 (0/0=0) (1/max=1), gen (obstproblem_cat6)
lab var obstproblem_cat6 "Experienced at least one obstetric problem"
lab def obstproblem_cat6 1 " Yes" 0 "No"
lab val obstproblem_cat6 obstproblem_cat6
ta obstproblem_cat6

*Survey seven
egen obstproblems7=rowtotal(m2_204a_et_r6 m2_204c_et_r6 m2_204f_et_r6 m2_204h_et_r6), mi
lab var obstproblems7 "Experience of obstetric problems"    

recode obstproblems7 (0/0=0) (1/max=1), gen (obstproblem_cat7)
lab var obstproblem_cat7 "Experienced at least one obstetric problem"
lab def obstproblem_cat7 1 " Yes" 0 "No"
lab val obstproblem_cat7 obstproblem_cat7
tab obstproblem_cat7 
				
*Survey eight
egen obstproblems8=rowtotal(m2_204a_et_r7 m2_204c_et_r7 m2_204f_et_r7 m2_204h_et_r7), mi
lab var obstproblems8 "Experience of obstetric problems" 

recode obstproblems8 (0/0=0) (1/max=1), gen(obstproblem_cat8)
lab var obstproblem_cat8 "Experienced at least one obstetric problem"
lab def obstproblem_cat8 1 " Yes" 0 "No"
lab val obstproblem_cat8 obstproblem_cat8
tab obstproblem_cat8

**Women's experience of pregnancy danger signs at different survey points from m1 to m2 ( s1_dangersignm1 to s8_dangersignm2)**
*Survey one
egen pxdangers1=rowtotal(m1_814a m1_814b m1_814c m1_814d m1_814e m1_814f ///
m1_814g m1_814h m1_814i), mi
lab var pxdangers1 "Experience of pregnancy danger signs" 
ta pxdangers1
    
gen pxdanger_cat1=pxdangers1
recode pxdanger_cat1 (0=0) (1/max=1)
lab var pxdanger_cat1 "Experienced at least one pregnancy danger sign"
lab def pxdanger_cat1 1 " Yes" 0 "No"
lab val pxdanger_cat1 pxdanger_cat1

*Survy2
egen pxdangers2=rowtotal(m2_203a_r1 m2_203b_r1 m2_203c_r1 m2_203d_r1 ///
 m2_203e_r1 m2_203f_r1 m2_203g_r1 m2_203h_r1 m2_203i_r1), mi 
lab var pxdangers2 "Experience of pregnancy danger signs"
ta pxdangers2
    
gen pxdanger_cat2=pxdangers2
recode pxdanger_cat2 (0=0) (1/max=1)
lab var pxdanger_cat2 "Experienced at least one pregnancy danger sign"
lab def pxdanger_cat2 1 " Yes" 0 "No"
lab val pxdanger_cat2 pxdanger_cat2
tab pxdanger_cat2 

*Survey3
egen pxdangers3=rowtotal(m2_203a_r2 m2_203b_r2 m2_203c_r2 m2_203d_r2 ///
 m2_203e_r2 m2_203f_r2 m2_203g_r2 m2_203h_r2 m2_203i_r2), mi
lab var pxdangers3 "Experience of pregnancy danger signs"
ta pxdangers3
    
gen pxdanger_cat1=pxdangers3
recode pxdanger_cat1 (0=0) (1/max=1)
lab var pxdanger_cat1 "Experienced at least one pregnancy danger sign"
lab def pxdanger_cat1 1 " Yes" 0 "No"
lab val pxdanger_cat1 pxdanger_cat1
ta pxdanger_cat1

*survey4
egen pxdangers4=rowtotal(m2_203a_r3 m2_203b_r3 m2_203c_r3 m2_203d_r3 ///
 m2_203e_r3 m2_203f_r3 m2_203g_r3 m2_203h_r3 m2_203i_r3), mi 
lab var pxdangers4 "Experience of pregnancy danger signs"
ta pxdangers4
    
gen pxdangers4_cat=pxdangers4
recode pxdangers4_cat (0=0) (1/max=1)
lab var pxdangers4_cat "Experienced at least one pregnancy danger sign"
lab def pxdangers4_cat 1 " Yes" 0 "No"
lab val pxdangers4_cat pxdangers4_cat
tab pxdangers4_cat
rename pxdangers4_cat pxdanger_cat4
		
*Survey5
 egen pxdangers5=rowtotal(m2_203a_r4 m2_203b_r4 m2_203c_r4 m2_203d_r4 ///
 m2_203e_r4 m2_203f_r4 m2_203g_r4 m2_203h_r4 m2_203i_r4), mi
 lab var pxdangers5 "Experience of pregnancy danger signs"
 tab pxdangers5
    
 gen pxdanger_cat5=pxdangers5
 recode pxdanger_cat5 (0=0) (1/max=1)
 lab var pxdanger_cat5 "Experienced at least one pregnancy danger sign"
 lab def pxdanger_cat5 1 " Yes" 0 "No"
 lab val pxdanger_cat5 pxdanger_cat5
 tab pxdanger_cat5
		
**Persistent experience of pregnancy danger signs**
egen pers_pxdanger_s1s4=rowtotal(pxdanger_cat1 pxdanger_cat2 pxdanger_cat3 pxdanger_cat4), mi
lab var pers_pxdanger_s1s4 "Persistently experienced at least one pregnancy danger sign across 4 survey points"
tab pers_pxdanger_s1s4 
		
egen pers_pxdanger_s1s5=rowtotal(ppxdanger_cat1 pxdanger_cat2 ///
pxdanger_cat3 pxdanger_cat4 pxdangers5_cat), mi
lab var pers_pxdanger_s1s5 "Persistently experienced at least one pregnancy danger sign across 5 survey points"
tab pers_pxdanger_s1s5	

*//Survey6
 egen pxdangers6=rowtotal(m2_203a_r5 m2_203b_r5 m2_203c_r5 m2_203d_r5 ///
 m2_203e_r5 m2_203f_r5 m2_203g_r5 m2_203h_r5 m2_203i_r5), mi
 lab var pxdangers6 "Experience of pregnancy danger signs"
 
 gen pxdanger_cat6=pxdangers6
 recode pxdanger_cat6 (0=0) (1/max=1)
 lab var pxdanger_cat6 "Experienced at least one pregnancy danger sign"
 lab def pxdanger_cat6 1 " Yes" 0 "No"
 lab val pxdanger_cat6 pxdanger_cat6
 tab pxdanger_cat6

*Survey7
egen pxdangers7=rowtotal(m2_203a_r6 m2_203b_r6 m2_203c_r6 m2_203d_r6 ///
m2_203e_r6 m2_203f_r6 m2_203g_r6 m2_203h_r6 m2_203i_r6), mi
lab var pxdangers7 "Experience of pregnancy danger signs" 
 
gen pxdanger_cat7=pxdangers7
recode pxdanger_cat7 (0=0) (1/max=1)
lab var pxdanger_cat7 "Experienced at least one pregnancy danger sign"
lab def pxdanger_cat7 1 " Yes" 0 "No"
lab val pxdanger_cat7 pxdanger_cat7
tab pxdanger_cat7

*Survey8
egen pxdangers8=rowtotal(m2_203a_r7 m2_203b_r7 m2_203c_r7 m2_203d_r7 ///
m2_203e_r7 m2_203f_r7 m2_203g_r7 m2_203h_r7 m2_203i_r7), mi
lab var pxdangers8 "Experience of pregnancy danger signs"

gen pxdanger_cat8=pxdangers8
recode pxdanger_cat8 (0=0) (1/max=1)
lab var pxdanger_cat8 "Experienced at least one pregnancy danger sign"
lab def pxdanger_cat8 1 " Yes" 0 "No"
lab val pxdanger_cat8 pxdanger_cat8
tab pxdanger_cat8

**Women with ANC visit,
   * First, create a new variable for the first visit of ANC for all women at enrollment, taken as all taken 
 gen m1_anc1_r1=facility_type
 recode m1_anc1_r1 (1/4=1) (.=0)
 lab var m1_anc1_r1 "ANC visit at enrollment" 
 lab def m1_anc1_r1  1 "Yes" 0 "No"
 lab val m1_anc1_r1 m1_anc1_r1
 tab m1_anc1_r1

 tab m2_301_r1  
 tab m2_301_r2
 tab m2_301_r3
 tab m2_301_r4
 tab m2_301_r5
 tab m2_301_r6
 tab m2_301_r7
 egen anc_visit=rowtotal (m1_anc1_r1 m2_301_r1 m2_301_r2 m2_301_r3 ///
 m2_301_r4 m2_301_r5 m2_301_r6 m2_301_r7), mi
 lab var anc_visit "Number of ANC visit from m1 to end of m2"
 tab anc_visit

 {tab m2_302_r1 // Number of anc visit
 tab m2_302_r2    
 tab m2_302_r3    
 tab m2_302_r4}
 
 {tab m2_303a_r1 // Facility type that ANC visit 
 tab m2_303a_r2 
 tab m2_303a_r3 
 tab m2_303a_r4}
			
*Pregnant women who experienced a pregnancy danger sign and received counseling for a pregnancy danger sign at d/t time of their ANC visit

  //Note: Consideration for pregnancy danger sign counseling service for those mothers experienced any pregnancy danger signs
		//m2-506a_r1-8 was asked for all the PX women who had an ANC visit. As a result, we can generate a new variable "how many of the pregnant women who received pregnancy danger sign counseling among those who experienced at least one pregnancy danger sign from the HCWs at their ANC visit/m2_506a_r1=1/yes". 
  // To generate this new variable, the following variables will be considered: m2_301_r1=1/yes & pregnant women experienced at least one danger sign (pxdangers2_cat =1/yes) and  HCWs discussed about pregnancy danger sign with at time of ANC visit (m2_506_r1 =1/yes) coded as 1/Yes while m2_301_r1=1 & pxdangers2_cat==1 & m2_506_r1 ==0 coded as 0/No
  
**Women received counseling for pregnancy danger signs among those who experienced any danger signs and received ANC
rename (pxdangers1_cat pxdangers2_cat pxdangers3_cat pxdangers4_cat)  pxdangers1 pxdangers2 pxdangers3 pxdangers4)
*Survey one:  
gen counspxdangers1=.
replace counspxdangers1=1 if pxdangers1_cat==1 & m1_716e==1
replace counspxdangers1=0 if pxdangers1_cat==1 & m1_716e==0
lab var counspxdangers1 "Women with Px danger sign received counseling from HCWs"
lab def counspxdangers1 1 "Received" 0 "Not received"
lab val counspxdangers1 counspxdangers1
tab counspxdangers1
	 
*Survey two: women received counseling for pregnancy danger signs, among those who experienced any danger signs and received ANC
gen counspxdangers2=.
replace counspxdangers2=1 if pxdangers2_cat==1 & m2_301_r1==1 & m2_506a_r1==1 
replace counspxdangers2=0 if pxdangers2_cat==1 & m2_301_r1==1 & m2_506a_r1==0
lab var counspxdangers2 "Women with Px danger sign received counseling from HCWs" 
lab def counspxdangers2 1 "Received" 0 "Not received"
lab val counspxdangers2 counspxdangers2
tab  counspxdangers2
	
**Survey three: women received counseling for pregnancy danger signs, among those who experienced any danger signs and received ANC
gen counspxdangers3=.
replace counspxdangers3=1 if pxdangers3_cat==1 & m2_301_r2==1 & m2_506a_r2==1 
replace counspxdangers3=0 if pxdangers3_cat==1 & m2_301_r2==1 & m2_506a_r2==0
lab var counspxdangers3 "Women with Px danger sign received counseling from HCWs" 
lab def counspxdangers3 1 "Received" 0 "Not received"
lab val counspxdangers3 counspxdangers3
tab counspxdangers3
		
*Survey four: women received counseling for pregnancy danger signs, among those who experienced any danger signs and received ANC

gen counspxdangers4=.
replace counspxdangers4=1 if pxdangers4_cat==1 & m2_301_r3==1 & m2_506a_r3==1 
replace counspxdangers4=0 if pxdangers4_cat==1 & m2_301_r3==1 & m2_506a_r3==0
lab var counspxdangers4 "Women with Px danger sign received counseling from HCWs" 
lab def counspxdangers4 1 "Received" 0 "Not received"
lab val counspxdangers4 counspxdangers4
tab counspxdangers4
	
*Survey five: women received counseling for pregnancy danger signs, among those who experienced any danger signs and received ANC

gen counspxdangers5=.
replace counspxdangers5=1 if pxdangers5_cat==1 & m2_301_r4==1 & m2_506a_r4==1 
replace counspxdangers5=0 if pxdangers5_cat==1 & m2_301_r4==1 & m2_506a_r4==0
lab var counspxdangers5 "Women with Px danger sign received counseling from HCWs" 
lab def counspxdangers5 1 "Received" 0 "Not received"
lab val counspxdangers5 counspxdangers5
tab counspxdangers5
	
**Persistent counseling of pregnancy danger signs for those experienced danger signs*
egen pers_conspxdanger_s1s4=rowtotal(counspxdangers1 counspxdangers2 counspxdangers3 counspxdangers4), mi
lab var pers_conspxdanger_s1s4 "Consistently counseled for pregnancy danger signs"
tab pers_conspxdanger_s1s4 
		
egen pers_conspxdanger_s1s5=rowtotal (counspxdangers1 counspxdangers2 ///
 counspxdangers3 counspxdangers4 counspxdangers5), mi
lab var pers_conspxdanger_s1s5 "Consistently counseled for pregnancy danger signs" 
tab pers_conspxdanger_s1s5 
	
* Survey six: women received counseling for pregnancy danger signs, among those who experienced any danger signs and received ANC
 gen counspxdangers6=.
 replace counspxdangers6=1 if pxdanger_cat6==1 & m2_301_r5==1 & m2_506a_r5==1 
 replace counspxdangers6=0 if pxdanger_cat6==1 & m2_301_r5==1 & m2_506a_r5==0
 lab var counspxdangers6 "Women with Px danger sign received counseling from HCWs" 
 lab def counspxdangers6 1 "Received" 0 "Not received"
 lab val counspxdangers6 counspxdangers6
 tab counspxdangers6
	
*Survey seven: women received counseling for pregnancy danger signs, among those who experienced any danger signs and received ANC
 gen counspxdangers7=.
 replace counspxdangers7=1 if pxdanger_cat7==1 & m2_301_r6==1 & m2_506a_r6==1 
 replace counspxdangers7=0 if pxdanger_cat7==1 & m2_301_r6==1 & m2_506a_r6==0
 lab var counspxdangers7 "Women with Px danger sign received counseling from HCWs" 
 lab def counspxdangers7 1 "Received" 0 "Not received"
 lab val counspxdangers7 counspxdangers7
 ta counspxdangers7
 
*Survey eight: women received counseling for pregnancy danger signs, among those who experienced any danger signs and received ANC
 gen counspxdangers8=.
 replace counspxdangers8=1 if pxdanger_cat8==1 & m2_301_r7==1 & m2_506a_r7==1 
 replace counspxdangers8=0 if pxdanger_cat8==1 & m2_301_r7==1 & m2_506a_r7==0
 lab var counspxdangers8 "Women with Px danger sign received counseling from HCWs" 
 lab def counspxdangers8 1 "Received" 0 "Not received"
 lab val counspxdangers8 counspxdangers8
 tab counspxdangers8 
 
 {*//type of counseling service given for pregnant women on pregnancy danger signs
   tab m2_507_r1} // not relevant for knowing how many px women who experienced danger sign and received counseling for px danger signs 
   
********************************************************************************
***Women depressive symptoms during pregnancy and the recipient of care for depression at different survey points**//

**Survey one 
egen depressions1=rowtotal(m1_phq9a m1_phq9b m1_phq9c m1_phq9d m1_phq9e ///
 m1_phq9f m1_phq9g m1_phq9h m1_phq9i), mi
lab var depressions1 "Depressive symptoms score during pregnancy" 
tab depressions1
		
gen depre_cat1=depressions1
recode depre_cat1 (0/4=0) (5/max=1)
lab var depre_cat1 "Women experienced depressive symptoms during pregnancy"
lab def depre_cat1 1 " Experienced" 0 "Not Experienced"
lab val depre_cat1 depre_cat1
tab depre_cat1
		
*Women's depressive symptoms considering the 8 cutoff points 
gen depre8cut_cat1=depressions1  // only generated in 1000, m1 to m2_r4 936 and 728 wide dataset
recode depre8cut_cat1 (0/7=0) (8/max=1)
lab var depre8cut_cat1 "Women experienced depressive symptoms with ≥ 8-point"
lab def depre8cut_cat1 1 " Experienced" 0 "Not Experienced"
lab val depre8cut_cat1 depre8cut_cat1
tab depre8cut_cat1

*Level of women's depressive symptoms 
gen depre_level1=depressions1  // only generated for wide data set. so please generate for long data set
recode depre_level1 (0/4=1) (5/9=2) (10/14=3) (15/max=4)
lab var depre_level1 "Women's level of depressive symptoms during pregnancy"
lab def depre_level1 1 "Normal 0-4"  2 "Mild 5-9" 3 " Moderate 10-14" 4 "Moderately severe or more 15-27"
lab val depre_level1 depre_level1
tab depre_level1

**Survey two  
egen depressions2=rowtotal(m2_205a_r1 m2_205b_r1 m2_205c_r1 m2_205d_r1 ///
 m2_205e_r1 m2_205f_r1 m2_205g_r1 m2_205h_r1 m2_205i_r1), mi
lab var depressions2 "Depressive symptoms score during pregnancy"
tab depressions2
    
gen deprs2_5cutoff=depressions2
recode deprs2_5cutoff (0/4=0) (5/max=1)
lab var deprs2_5cutoff "Women experienced depressive symptoms during pregnancy"
lab def deprs2_5cutoff 1 "Experienced" 0 "Not Experienced"
lab val deprs2_5cutoff deprs2_5cutoff
tab deprs2_5cutoff
rename deprs2_5cutoff depre_cat2	

*Women's depressive symptoms considering the 8 cutoff points 
gen deprs2_8cutoff=depressions2  // only generated in 1000, m1 to m2_r4 936 and 728 wide dataset
recode deprs2_8cutoff (0/7=0) (8/max=1)
lab var deprs2_8cutoff "Women experienced depressive symptoms with ≥ 8-point"
lab def deprs2_8cutoff 1 "Experienced" 0 "Not Experienced"
lab val deprs2_8cutoff deprs2_8cutoff
tab deprs2_8cutoff
rename deprs2_8cutoff depre8cut_cat2	
 
*Level of women's depressive symptoms 
gen deprs2_4cat=depressions2  // only generated for wide data set. So please generate for other wide and long datasets
recode deprs2_4cat (0/4=1) (5/9=2) (10/14=3) (15/max=4)
lab var deprs2_4cat "Women's level of depressive symptoms during pregnancy"
lab def deprs2_4cat 1 "Normal 0-4"  2 "Mild 5-9" 3 " Moderate 10-14" 4 "Moderately severe or more 15-27"
lab val deprs2_4cat deprs2_4cat
tab deprs2_4cat
rename deprs2_4cat depre_level2

bro id m2_205a_r1 m2_205b_r1 m2_205c_r1 m2_205d_r1 m2_205e_r1 m2_205f_r1 ///
 m2_205g_r1 m2_205h_r1 m2_205i_r1 depr_cat2 depre_level2
   *three women with 180838/row-934,241927/row_935 and 100725/row-936 
   *had a missed depression measurements at survey two  

**Survey three 
egen depressions3=rowtotal(m2_205a_r2 m2_205b_r2 m2_205c_r2 m2_205d_r2 ///
 m2_205e_r2 m2_205f_r2 m2_205g_r2 m2_205h_r2 m2_205i_r2), mi
lab var depressions3 "Pregnant women's depression score" 
tab depressions3
		
gen depre_cat3=depressions3
recode depre_cat3 (0/4=0) (5/max=1)
lab var depre_cat3 "Women experienced depressive symptoms during pregnancy"
lab def depre_cat3 1 "Experienced" 0 "Not Experienced"
lab val depre_cat3 depre_cat3
tab depre_cat3

*Women's depressive symptoms considering the 8 cutoff points 
gen depre8cut_cat3=depressions3
recode depre8cut_cat3 (0/7=0) (8/max=1)
lab var depre8cut_cat3 "Women experienced depressive symptoms with ≥ 8-point"
lab def depre8cut_cat3 1 "Experienced" 0 "Not Experienced"
lab val depre8cut_cat3 depre8cut_cat3
tab depre8cut_cat3

*Level of women's depressive symptoms 
gen depre_level3=depressions3 
recode depre_level3 (0/4=1) (5/9=2) (10/14=3) (15/max=4)
lab var depre_level3 "Women's level of depressive symptoms during pregnancy"
lab def depre_level3 1 "Normal 0-4"  2 "Mild 5-9" 3 " Moderate 10-14" ///
 4 "Moderately severe or more 15-27"
lab val depre_level3 depre_level3
tab depre_level3
			 
bro id m2_205a_r2 m2_205b_r2 m2_205c_r2 m2_205d_r2 m2_205e_r2 m2_205f_r2 ///
  m2_205g_r2 m2_205h_r2 m2_205i_r2 depre_cat3 depre_level3
*The mother with id "110536/row" did not measure her depressive symptoms at survey three or m2_r2; she may be absent at this interview		 
			  
**Survey four  
egen depressions4=rowtotal(m2_205a_r3 m2_205b_r3 m2_205c_r3 m2_205d_r3 ///
 m2_205e_r3 m2_205f_r3 m2_205g_r3 m2_205h_r3 m2_205i_r3), mi
lab var depressions4 "Depressive symptoms score during pregnancy" 
tab depressions4
    
gen depre_cat4=depressions4
recode depre_cat4 (0/4=0) (5/max=1)
lab var depre_cat4 "Women experienced depressive symptoms"
lab def depre_cat4 1 "Experienced" 0 "Not Experienced"
lab val depre_cat4 depre_cat4
tab depre_cat4
		
*Women's depressive symptoms considering the 8 cutoff points 
gen depre8cut_cat4=depressions4  
recode depre8cut_cat4 (0/7=0) (8/max=1)
lab var depre8cut_cat4 "Women experienced depressive symptoms with ≥ 8-point"
lab def depre8cut_cat4 1 "Experienced" 0 "Not Experienced"
lab val depre8cut_cat4 depre8cut_cat4
tab  depre8cut_cat4
 
*Level of women's depressive symptoms 
gen depre_level4=depressions4  
recode depre_level4 (0/4=1) (5/9=2) (10/14=3) (15/max=4)
lab var depre_level4 "Women's level of depressive symptoms during pregnancy"
lab def depre_level4 1 "Normal 0-4"  2 "Mild 5-9" 3 " Moderate 10-14" ///
 4 "Moderately severe or more 15-27"
lab val depre_level4 depre_level4
tab depre_level4		

**Survey five 
egen depressions5=rowtotal(m2_205a_r4 m2_205b_r4 m2_205c_r4 m2_205d_r4 ///
m2_205e_r4 m2_205f_r4 m2_205g_r4 m2_205h_r4 m2_205i_r4), mi
lab var depressions5 "Depressive symptoms score during pregnancy" 
	
gen depre_cat5=depressions5
recode depre_cat5 (0/4=0) (5/max=1)
lab var depre_cat5 "Women experienced depressive symptoms during pregnancy"
lab def depre_cat5 1 "Experienced" 0 "Not Experienced"
lab val depre_cat5 depre_cat5
tab depre_cat5

*Women's depressive symptoms considering the 8 cutoff point 
gen depre8cut_cat5=depressions5 
recode depre8cut_cat5 (0/7=0) (8/max=1)
lab var depre8cut_cat5 "Women experienced depressive symptoms with ≥ 8-point" 
lab def depre8cut_cat5 1 "Experienced" 0 "Not Experienced"
lab val depre8cut_cat5 depre8cut_cat5
ta depre8cut_cat5
		 
*Level of women's depressive symptoms 
gen depre_level5=depressions5  
recode depre_level5 (0/4=1) (5/9=2) (10/14=3) (15/max=4)
lab var depre_level5 "Women's level of depressive symptoms during pregnancy"
lab def depre_level5 1 "Normal 0-4"  2 "Mild 5-9" 3 "Moderate 10-14" ///
4 "Moderately severe or more 15-27"
lab val depre_level5 depre_level5
ta depre_level5	
								
**Persistent experience of depressive symptoms during pregnancy from survey 1 to survey 5
*Considering a 5 cutoff point,
egen pers_dep_5cut_s1s4=rowtotal(depre_cat1 depre_cat2 depre_cat3 depre_cat4), mi
lab var pers_dep_5cut_s1s4 "Persistent experience of depressive symptoms" 
tab pers_dep_5cut_s1s4  

recode pers_dep_5cut_s1s4 0/1=0 2/max=1, gen (pers_depre_s1s4_cat)	
lab def pers_depre_s1s4_cat  1 " Experienced twice and more"  0 "Experienced < twice" 
lab val pers_depre_s1s4_cat pers_depre_s1s4_cat	

egen pers_dep_5cut_s1s3=rowtotal(depre_cat1 depre_cat2 depre_cat3), mi
lab var pers_dep_5cut_s1s3 "Persistent experience of depressive symptoms" 
tab pers_dep_5cut_s1s3
 
recode pers_dep_5cut_s1s3 0/1=0 2/max=1, gen (pers_depre_s1s3_cat)	
lab def pers_depre_s1s3_cat  1 " Experienced twice and more"  0 "Experienced < twice" 
lab val pers_depre_s1s3_cat pers_depre_s1s3_cat		
	
egen pers_dep_5cut_s1s5=rowtotal(epre_cat1 depre_cat2 depre_cat3 depre_cat4 deprs5_5cutoff ), mi
lab var pers_dep_5cut_s1s5 "Persistent experience of depressive symptoms"
tab pers_dep_5cut_s1s5  

*Considering 8 cutoff point,
egen pers_dep_8cut_s1s4=rowtotal(depre8cut_cat1 depre8cut_cat2 depre8cut_cat3 depre8cut_cat4 ), mi  
lab var pers_dep_8cut_s1s4 "Persistent experience of depressive symptoms" 
tab pers_dep_8cut_s1s4  //Only generated in 1000, m1 to m2_r4 936  and 728 wide dataset
			  
egen pers_dep_8cut_s1s5=rowtotal(depre8cut_cat1 depre8cut_cat2 depre8cut_cat3 depre8cut_cat4 depre8cut_cat5),mi 
lab var pers_dep_8cut_s1s5 "Persistent experience of depressive symptoms"
tab pers_dep_8cut_s1s5 //Only generated in 1000, m1 to m2_r4 936 and 507 wide dataset 
				
*//Survey six 
egen depressions6=rowtotal(m2_205a_r5 m2_205b_r5 m2_205c_r5 m2_205d_r5 ///
m2_205e_r5 m2_205f_r5 m2_205g_r5 m2_205h_r5 m2_205i_r5), mi
lab var depressions6  "Depressive symptoms score during pregnancy" 
  
gen depre_cat6=depressions6
recode depre_cat6 (0/4=0) (5/max=1)
lab var depre_cat6 "Women experienced depressive symptoms during pregnancy"
lab def depre_cat6 1 "Experienced" 0 "Not Experienced"
lab val depre_cat6 depre_cat6
tab depre_cat6

*Women's depressive symptoms considering the 8 cutoff points 
gen depre8cut_cat6=depressions6  // only derived for 1000 wide data set
recode depre8cut_cat6 (0/7=0) (8/max=1)
lab var depre8cut_cat6  "Women experienced depressive symptoms with ≥ 8-point"
lab def depre8cut_cat6 1 "Experienced" 0 "Not Experienced"
lab val depre8cut_cat6 depre8cut_cat6
ta depre8cut_cat6
			 
*Level of women's depressive symptoms 
 gen depre_level6=s6_depressionm2  // only generated for 1000 wide data set. So, please generate for other wide and long datasets
recode depre_level6 (0/4=1) (5/9=2) (10/14=3) (15/max=4)
lab var depre_level6 "Women's level of depressive symptoms during pregnancy"
lab def depre_level6 1 "Normal 0-4"  2 "Mild 5-9" 3 "Moderate 10-14" 4 "Moderately severe or more 15-27"
lab val depre_level6 depre_level6
ta depre_level6
		
*Survey Seven 
egen depressions7=rowtotal(m2_205a_r6 m2_205b_r6 m2_205c_r6 m2_205d_r6 ///
m2_205e_r6 m2_205f_r6 m2_205g_r6 m2_205h_r6 m2_205i_r6), mi
lab var depressions7  "Depressive symptoms score during pregnancy" 
rename s7_depressionm2 depressions7
 
gen depre_cat7=depressions7
recode depre_cat7 (0/4=0) (5/max=1)
lab var depre_cat7 "Women experienced of depressive symptoms during pregnancy" 
lab def depre_cat7 1 "Experienced" 0 "Not Experienced"
lab val depre_cat7 depre_cat7
 
*Women's depressive symptoms considering the 8 cutoff points 
gen depre8cut_cat7=depressions7 // only derived for 1000 wide data set
recode depre8cut_cat7 (0/7=0) (8/max=1)
lab var depre8cut_cat7  "Women experienced depressive symptoms with ≥ 8-point"
lab def depre8cut_cat7 1 "Experienced" 0 "Not Experienced"
lab val depre8cut_cat7 depre8cut_cat7
ta depre8cut_cat7

*Level of women's depressive symptoms 
gen depre_level7=depressions7  
recode depre_level7 (0/4=1) (5/9=2) (10/14=3) (15/max=4)
lab var depre_level7 "Women's level of depressive symptoms during pregnancy" 
lab def depre_level7 1 "Normal 0-4"  2 "Mild 5-9" 3 "Moderate 10-14" 4 "Moderately severe or more 15-27"
lab val depre_level7 depre_level7
ta depre_level7 

*Survey eight
egen depressions8=rowtotal(m2_205a_r7 m2_205b_r7 m2_205c_r7 m2_205d_r7 ///
 m2_205e_r7 m2_205f_r7 m2_205g_r7 m2_205h_r7 m2_205i_r7), mi
lab var depressions8 "Depressive symptoms score during pregnancy" 

gen depre_cat8=depressions8
recode depre_cat8 (0/4=0) (5/max=1)
lab var depre_cat8 "Women experienced of depressive symptoms during pregnancy" 
lab def depre_cat8 1 "Experienced" 0 "Not Experienced"
lab val depre_cat8 depre_cat8
ta depre_cat8

*Women's depressive symptoms considering the 8 cutoff points 
gen depre8cut_cat8=depressions8 // only derived for 1000 wide data set
recode depre8cut_cat8 (0/7=0) (8/max=1)
lab var depre8cut_cat8  "Women experienced depressive symptoms with ≥ 8-point"
lab def depre8cut_cat8 1 "Experienced" 0 "Not Experienced"
lab val depre8cut_cat8 depre8cut_cat8
ta depre8cut_cat8
	
*Level of women's depressive symptoms 
gen depre_level8=depressions8  
recode depre_level8 (0/4=1) (5/9=2) (10/14=3) (15/max=4)
lab var depre_level8 "Women's level of depressive symptoms during pregnancy"
lab def depre_level8 1 "Normal 0-4" 2 "Mild 5-9" 3 "Moderate 10-14" 4 "Moderately severe or more 15-27"
lab val depre_level8 depre_level8
tab depre_level8 

******************************************************************
 The recipient of care for depression/mental health service
*****************************************************************
*The recipient of care for baseline includes: 
  // The HCWs or the woman discussed with anxiety or depression (m1_716c or m1_717) OR 
  // The provider instructed to get a mental health care provider/ psychologist(m1_724d) OR 
  // The provider gave/prescribed medications for mental health problems (m1_713f) 
"Following these: a pregnant woman who received at least one of these services 
// (m1_716c or m1_717, m1_724d, m1_713f) during their contact consideded as having optimal care fro depressive /mental health service"
  
*While the recipient of care for follow-up includes: 
// Pregnant women had a session of psychological counseling or therapy with any type of professionals (m2_508a_r1, r_2...r_8):
   *Including seeing a mental health professional (psychologist, social worker, nurse, spiritual advisor, or healer)
   
*Survey one
*First, recode the variable "m1_713f" into a binary outcome (Yes, No)
gen m1_713f_medic_emotion=m1_713f
recode m1_713f_medic_emotion (1/2=1) (3=0)
lab var m1_713f_medic_emotion "Gave or prescribed medication for mental health" //id 71523/r-466 corrected as "0" instead of missing
lab def m1_713f_medic_emotion 1 "Yes" 2 "No"
tab m1_713f_medic_emotion
		 
{gen s1_depression_mat=.
replace s1_depression_mat=1 if m1_716c==1|m1_717==1|m1_724d==1|m1_713f_medic_emotion==1
lab var s1_depression_mat "number of women who received MAT for depression"
drop s1_depression_mat} 
		   *OR*
*Survey one: recipient of care
egen depcares1=rowtotal(m1_716c m1_717 m1_724d m1_713f_medic_emotion), mi
lab var depcares1 "The recipient of care for depressive symptoms"
tab depcares1
	
gen deprecare_cat1=depcares1
recode deprecare_cat1 (0=0) (1/max=1)
lab var deprecare_cat1 "Recipient of care receivedfor depressive symptoms"
lab def deprecare_cat1 1 "Optimal care" 0 "Non-optimal care" 
lab val deprecare_cat1 deprecare_cat1
tab deprecare_cat1

//received optimal care indicated a woman who received at least one care out of four cares	
*Survey two	
gen deprecare_cat2= m2_508a_r1
recode deprecare_cat2 (1=1) (0=0)
lab var deprecare_cat2 "Recipient of care received for depressive symptoms"   
lab def deprecare_cat2 1 "Optimal care" 0 "Non-optimal care" 
lab val deprecare_cat2 deprecare_cat2
tab deprecare_cat2

*Survey three
gen deprecare_cat3= m2_508a_r2
recode deprecare_cat3 (1=1) (0=0)
lab var deprecare_cat3 "Recipient of care received for depressive symptoms"  
lab def deprecare_cat3 1 "Optimal care" 0 "Non-optimal care" 
lab val deprecare_cat3 deprecare_cat3
tab deprecare_cat3
 //A woman received psychological counseling/therapy from any HCWs 
 
*Survey four
gen deprecare_cat4= m2_508a_r3
recode deprecare_cat4 (1=1) (0=0)
lab var deprecare_cat4 "Recipient of care received for depressive symptoms" 
lab def deprecare_cat4 1 "Optimal care" 0 "Non-optimal care"
lab val deprecare_cat4 deprecare_cat4
tab deprecare_cat4
 //A woman received psychological counseling/therapy from any HCWs 
 
*Survey five
 gen deprecare_cat5= m2_508a_r4
 recode deprecare_cat5 (1=1) (0=0)
 lab var deprecare_cat5 "Recipient of care received for depressive symptoms"  
 lab def deprecare_cat5 1 "Optimal care" 0 "Non-optimal care"
 lab val deprecare_cat5 deprecare_cat5
		
//Survey six
gen deprecare_cat6= m2_508a_r5
recode deprecare_cat6 (1=1) (0=0)
lab var deprecare_cat6 "Recipient of care received for depressive symptoms"  
lab def deprecare_cat6 1 "Optimal care" 0 "Non-optimal care"
lab val deprecare_cat6 deprecare_cat6

		
*Survey seven
gen deprecare_cat7= m2_508a_r6
recode deprecare_cat7 (1=1) (0=0)
lab var deprecare_cat7 "Recipient of care received for depressive symptoms"  
lab def deprecare_cat7 1 "Optimal care" 0 "Non-optimal care"
lab val deprecare_cat7 deprecare_cat7
		
*Survey eight
gen deprecare_cat8= m2_508a_r7
recode deprecare_cat8 (1=1) (0=0)
lab var deprecare_cat8 "Recipient of care received for depressive symptoms"  
lab def deprecare_cat8 1 "Optimal care" 0 "Non-optimal care"
lab val deprecare_cat8 deprecare_cat8 
	
******************************************************************************
"Survey time difference between each measurement point"
******************************************************************************
gen survtimebn1_1=(m2_date - m1_date)/7 // survey time b/n measurement one
lab var survtimebn1_1 "Survey time at measurement one in weeks"
tab survtimebn1_1
	*done by excel & imported into STATA, 
gen survtimebn1_2=(m2_date_r1 - m1_date)/7 //Survey time b/n measurement 1 &2 
lab var survtimebn1_2 "Survey time between measurement 1 & 2 in weeks"
tab survtimebn1_2
	
gen survtimebn2_3=(m2_date_r2 - m2_date_r1)/7 //Survey time b/n measurement 2&3 
lab var survtimebn2_3 "Survey time between measurement 2 & 3 in weeks"
tab survtimebn2_3

gen survtimebn3_4=(m2_date_r3 -m2_date_r2)/7 // Survey time between measurement 3 & 4 
lab var survtimebn3_4 "Survey time between measurement 3 & 4 in weeks"
replace survtimebn3_4=round(survtimebn3_4,0.1)
ta survtimebn3_4
	
gen survtimebn4_5=(m2_date_r4 -m2_date_r3)/7 // Survey time between measurement 4 and 5
lab var survtimebn4_5 "Survey time between measurement 4 & 5  in weeks"
tab survtimebn4_5

gen survtimebn5_6=(m2_date_r5 -m2_date_r4)/7 // Survey time between measurement 4 and 5
lab var survtimebn5_6 "Survey time between measurement 5 & 6  in weeks"
tab survtimebn5_6

gen survtimebn6_7=(m2_date_r6 -m2_date_r5)/7 // Survey time between measurement 6 and 7
lab var survtimebn6_7 "Survey time between measurement 6 & 7  in weeks"
tab survtimebn6_7

gen survtimebn7_8=(m2_date_r7 -m2_date_r6)/7 // Survey time between measurement 7 and 8
lab var survtimebn7_8 "Survey time between measurement 7 & 8  in weeks"
tab survtimebn7_8
order survtimebn5_6 survtimebn6_7 survtimebn7_8, after(survtimebn4_5)
 *NB: survey time diffrence was computed by Excel & imported into STATA due to date format problem 
 
**The survey duration of women from enrollment (survey one/m1_date) to each consecutive measurement/survey**
*Survey time at enrollment/survey one (m1_date) and measurement one/survey one (survdurations1) in weeks
  // NB"womens suirvey duration was computed using Excel 
  // due to date format issues for generating the variable
gen survdurations1= (m1_date-m1_date)/7, missing  
lab var survdurations1 "Survey duration from enrollment to measurement point in weeks"
tab survdurations1
	
*Survey duration from enrollment/survey one (m1_date) to measurement two/survey two 
gen survdurations2= (m2_date_r1 - m1_date)/7, mi  
lab var survdurations2 "Survey duration from enrollment to measurement point in weeks"
tab survdurations2
	
*Survey time b/n enrollment/survey one and measurement three/survey three 
gen survdurations3= (m2_date_r2 - m1_date)/7, mi
lab var survdurations3 "Survey duration from enrollment to measurement point in weeks"
tab survdurations3
		 
*Survey time b/n enrollment/survey one and measurement three/survey four 
gen survtime14=(m2_date_r3 - m1_date)/7, mi
lab var survdurations4 "Survey duration from enrollment to measurement point in weeks"
tab survdurations4

*Survey time b/n enrollment/survey one and measurement three/survey five 
gen survdurations5=(m2_date_r4 - m1_date)/7, mi
lab var survdurations5 "Survey duration from enrollment to measurement point in weeks"
tab survdurations5

*Survey time b/n enrollment/survey one and measurement three/survey six
gen survdurations6=(m2_date_r5 - m1_date)/7
lab var survdurations6 "Survey duration from enrollment to measurement point in weeks"
survdurations6	

*Survey time b/n enrollment/survey one and measurement three/survey seven
gen survdurations7=(m2_date_r6 - m1_date)/7, mi
lab var survdurations7 "Survey duration from enrollment to measurement point in weeks"
tab survdurations7  
 
*Survey time b/n enrollment/survey one and measurement three/survey eight
gen survdurations8=(m2_date_r7 - m1_date)/7, mi
lab var survdurations8 "Survey duration from enrollment to measurement point in weeks"
tab survdurations8
 
**Gestational age from enrollemnt to the end of pregnancy (ga_inrolled_pregnancy_ended)
gen ga_enrolled_pregnancy_ended=(m3_birth_or_ended-m1_date)/7
lab var ga_enrolled_pregnancy_ended "Gestational age of from enrollement to the end of pregnancy"
ta ga_enrolled_pregnancy_ended

**Gestational age and trimester level in each survey point
bro id m1_date m1_ga m1_trimester m1_ga1 m2_date_r1 m2_ga_r1 m2_date_r2 ///
m2_ga_r2 m2_date_r3 m2_ga_r3 m2_date_r4 m2_ga_r4
   
*Survey one gestational age and trimester level 
tab m1_ga
tab m1 trimester

gen gestationages1=m1_ga
lab var gestationages1 "Women's gestational age in weeks"
tab gestationages1
	   
gen trimesters1=gestationages1
lab var trimesters1 "Women's trimester level"
recode trimesters1 (min/12.9=1) (13/27.9=2) (28/max=3)
lab def trimesters1 1 "1st trimester 0-12 wks" 2 "2nd trimester 13-27 wks" 3 "3rd trimester 28-42 wks"
lab val trimesters1 trimesters1
tab trimesters1
	  
*Gestational age and trimester level survey two
  //NB: generated by excell due to date format issue
//First check the the pregnacy status of women using
//the geastational of the women determined only for stillpregnant at each survey time
 
bro idn m2_202_r1 depre_cat2 m1_ga  survdurations2 m2_ga_r1 gestationages2 trimesters2

egen gestationages2=rowtotal(m1_ga survdurations2), mi 
lab var gestationages2 "Women's gestational age in weeks" // based on m1_ga and survtime12
tab gestationages2

*Trimester level at survey two 
gen trimesters2= gestationages2
lab var trimesters2 "Women's trimester level"
recode trimesters2 (min/12.9=1) (13/27.9=2) (28/max=3)
lab def trimesters2 1 "1st trimester 0-12 wks" 2 "2nd trimester 13-27 wks" 3 "3rd trimester 28-42 wks"
lab val trimesters2 trimesters2
tab trimesters2
	
*Gestational age at survey three 
ta m2_202_r2
ta trimesters3
bro idn m2_202_r2 depre_cat3 m1_ga survdurations3 m2_ga_r2 gestationages3 trimesters3 // done data on Obj I data only
 
egen gestationages3= rowtotal(m1_ga survdurations3), mi
lab var gestationages3 "Women's gestational age in weeks" 
tab gestationages3
	 
*Trimester level at survey three 
gen trimesters3= gestationages3
lab var trimesters3 "Women's trimester level"
recode trimesters3 (min/12.9=1) (13/27.9=2) (28/max=3)
lab def trimesters3 1 "1st trimester 0-12 wks" 2 "2nd trimester 13-27 wks" 3 "3rd trimester 28-42 wks"
lab val trimesters3 trimesters3
tab trimesters3
	 
*Gestational age at survey four
ta m2_202_r3
ta trimesters4
bro idn m2_202_r3 depre_cat4 m1_ga survdurations4 m2_ga_r3 gestationages4 trimesters4 // done data on Obj I data only

egen gestationages4=rowtotal(m1_ga survdurations4), mi
lab var gestationages4 "Women's gestational age in weeks" 
tab gestationages4
 
*Trimester level at survey four 
gen trimesters4= gestationages4
lab var trimesters4 "Women's trimester level"
recode trimesters4 (min/12.9=1) (13/27.9=2) (28/max=3)
lab def trimesters4 1 "1st trimester 0-12 wks" 2 "2nd trimester 13-27 wks" 3 "3rd trimester 28-42 wks"
lab val trimesters4 trimesters4
tab trimesters4
	 	 
*Gestational age at survey five
ta m2_202_r4
ta trimesters5
bro idn m2_202_r4 depre_cat5 m1_ga survdurations5 m2_ga_r4 gestationages5 trimesters5 // done data on Obj I data only

egen gestationages5=rowtotal(m1_ga survdurations5), mi
lab var gestationages5 "Women's gestational age in weeks" 
tab gestationages5
 
*trimester level at survey five 
gen trimesters5= gestationages5
lab var trimesters5 "Women's trimester level"
recode trimesters5 (min/12.9=1) (13/27.9=2) (28/max=3)
lab def trimesters5 1 "1st trimester 0-12 wks" 2 "2nd trimester 13-27 wks" 3 "3rd trimester 28-42 wks"
lab val trimesters5 trimesters5
tab trimesters5

*Gestational age at survey six: Hold on this
ta m2_202_r5
ta trimesters6
bro idn m2_202_r5 depre_cat6 m1_ga survdurations6 m2_ga_r5 gestationages6 trimesters6 // done data on Obj I data only

egen gestationages6=rowtotal(m1_ga survdurations6),mi
lab var gestationages6 "Women's gestational age in weeks" 
replace gestationages6=round(gestationages6,0.1)
tab gestationages6

*trimester level at survey six 
gen trimesters6= gestationages6
lab var trimesters6 "Women's trimester level"
recode trimesters6 (min/12.9=1) (13/27.9=2) (28/max=3)
lab def trimesters6 1 "1st trimester 0-12 wks" 2 "2nd trimester 13-27 wks" 3 "3rd trimester 28-42 wks"
lab val trimesters6 trimesters6
ta  trimesters6

*Gestational age at survey seven 
ta m2_202_r6
ta trimesters7
bro idn m2_202_r6 depre_cat7 m1_ga survdurations7 m2_ga_r6 gestationages7 trimesters7 // done data on Obj I data only

egen gestationages7=rowtotal(m1_ga survdurations7),mi
lab var gestationages7 "Women's gestational age in weeks" 
replace gestationages7=round(gestationages7,0.1)
tab gestationages7

egen gestationages7=rowtotal(m1_ga survdurations7),mi
lab var gestationages7 "Women's gestational age in weeks" 
replace gestationages7=round(gestationages7,0.1)
tab gestationages7

*trimester level at survey seven
gen trimesters7= gestationages7
lab var trimesters7 "Women's trimester level"
recode trimesters7 (min/12.9=1) (13/27.9=2) (28/max=3)
lab def trimesters7 1 "1st trimester 0-12 wks" 2 "2nd trimester 13-27 wks" 3 "3rd trimester 28-42 wks"
lab val trimesters7 trimesters7
tab  trimesters7	

*Gestational age at survey eight 
ta m2_202_r7
ta anctrimesters8
bro idn m2_202_r7 ancdepre_cat8 m1_ga ancsurvdurations8 m2_ga_r7 ancgestationages8 anctrimesters8 // done data on Obj I data only

egen gestationages8=rowtotal(m1_ga survdurations8),mi
lab var gestationages8 "Women's gestational age in weeks" 
replace gestationages8=round(gestationages8,0.1)
tab gestationages8

*trimester level at survey seven
gen trimesters8= gestationages8
lab var trimesters8 "Women's trimester level"
recode trimesters8 (min/12.9=1) (13/27.9=2) (28/max=3)
lab def trimesters8 1 "1st trimester 0-12 wks" 2 "2nd trimester 13-27 wks" 3 "3rd trimester 28-42 wks"
lab val trimesters8 trimesters8
tab trimesters8
order	gestationages5 gestationages6 gestationages7 gestationages8, after(gestationages4)
order trimesters5 trimesters6 trimesters7 trimesters8, after (trimesters4)	
*****************************************************************************			
*------------------------------------------------------------------------------
  "The overall pregnancy outcome of module two (from m2_202_r1 to M2_202_r8"
*------------------------------------------------------------------------------
*A total of 890 women have known pregnancy outcomes**
*while 110 women were lost at several round of m2***** 
tab m2_202_r1
gen m2_202_overall_outcome=.
replace m2_202_overall_outcome=1 if m2_202_r1==2 | m2_202_r2==2 | m2_202_r3==2 | m2_202_r4==2 | m2_202_r5==2 | m2_202_r6==2 | m2_202_r7==2 | m2_202_r8==2
replace m2_202_overall_outcome=0 if m2_202_r1==3 | m2_202_r2==3 | m2_202_r3==3 | m2_202_r4==3 | m2_202_r5==3 | m2_202_r6==3 | m2_202_r7==3 | m2_202_r8==3
lab def m2_202_overall_outcome 1 "Delivered with alive" 0 "Something else happened"
lab var m2_202_overall_outcome "overall prgnancy outcomes"
lab val m2_202_overall_outcome m2_202_overall_outcome 
ta m2_202_overall_outcome
 bro id m2_202_r1 m2_202_r2 m2_202_r3 m2_date_r4 m2_202_r4 m2_date_r5 m2_202_r5 /// 
m2_date_r6 m2_202_r6 m2_date_r7 m2_202_r7 m2_date_r8 m2_202_r8 m3_date ///
m2_202_overall_outcome


******************************************************************************** 
             Data analysis for Objective one
"Title: Magnitude, trajectory and determinants of antenatal depressive symptoms"
********************************************************************************
*Before going to the analysis of long data, you have to keep your data focused on 
 // observations with completed data for specific variables first/
*For our first objective, the analysis should be based on completed depressive symptom observation at each time point*
keep if !missing(depre_cat, time) // for keeping the analysis based on completed observation of depressive symptoms and time
keep if !missing(depre_cat, time, id, other_var) // for keeping depression, time, id, and other variables completed observation
keep if !missing(depre_cat1, depre_cat2, depre_cat3, depre_cat4 depre_cat5) // for wide data

* to chek the missing vs completed observations
egen complete = rowmiss(depre_cat time)
tab complete // (0 vs 1); 1 indicate number of missed observations

*Then conduct the analysis based on completed observations (Complete case analysis)

*Descriptive statistics*
*Open the dataset from the objective one folder 
"open the wide data_sevent time points or six time points"
*Open the dataset from objective one folder 

*Compute the health literacy of pregnant women after reshaping the data 
 // from 4 category into 2 category
recode m1_health_literacy 1/2=1 3=2 4=3, gen(= m1_hliteracy)
lab def m1_hliteracy 1 "Poor/fair" 2 "Good" 3 "Very good"
lab val m1_hliteracy m1_hliteracy 
tab m1_hliteracy // not generated
  
*To show the label of each choice, you can run this command
"numlabel, add"

*Socio-demographic characteristics by residence of each variable with column percent
tab m1_age_cat1 m1_residence, col
tab preg_intent m1_residence, col
tab m1_anemia_px_edit_02 depr_5cutoff, chi2 row // row percentage of outcome vs exposure is very important for discussion
tab preg_intent          // to generate the frequency with percent
proportion preg_intent // to generate the proportion with a confidence interval
	  
tab m1_anemia_px // original data used this code
tab m1_anemia_px_edit // edit with odd ratio 7.2 vs 11.5 times (1000 vs 727 dataset)

*Compute the mean, SD, median and IQR of participants age
sum m1_enrollage, detail // for overall including Adama/urban and East Shewa/rural
	
keep if m1_residence==1 // for mean and IQR for Adama participant
sum m1_enrollage, detail
	
keep if m1_residence==2 // for mean and IQR for East Shewa zone participant
sum m1_enrollage, detail

*Average survey time interval between measurements in weeks
egen survtime_total=rowtotal (survtimebn1_2 survtimebn2_3 survtimenb3_4), missing
lab var survtime_total "Total survey time from survey 1 to 4"
ta survtime_total
gen survtime_average=survtime_total/3 // from 1 to 4 survey period
lab var survtime_average " the time interval between survey periods"
sum survtime_average // mean with SD of between survey periods
  
*Assumption checking 
  // The normality of data were check by using Histogram 
 // (Create a histogram of your variable, then If it looks roughly bell-shaped and 
 // symmetric → it's likely close to normal), 
 // Q-Q plot (Plots your data's quantiles against a theoretical normal distribution, 
 // then If points lie close to the diagonal line → data is approximately normal) and 
 // Box plot (Can show skewness and outliers, Dots outside whiskers → outliers  then 
 // Strong skew or many outliers → likely not normal) graphically and

 
histogram depression, normal freq ///
title("Distribution of Depression Scores") ///
xtitle("Depression Score") ytitle("Frequency") ///
lcolor(red) fcolor(gs12)  // Histogram with normal curve

histogram depression, normal percent title("Distribution of Depression Scores") xtitle("Depression Score") ytitle("Percentage") fcolor(gs12) // in percentage

graph bar depression pxdangers anc_hrqol_score, ///
title("Mean of Psychological Variables") ///
ytitle("Mean Score") // showing diffrent variable in one bar grpah

graph bar depression pxdangers anc_hrqol_score, over(time) ///
title("Mean Scores by Group") ///
ytitle("Mean Value") // by survey time



qnorm depression              // Q–Q plot
graph box depression          // box plot
graph box depre_cat depre8cut_cat depre_level // multiple variable
graph box income, over(gender) // presenting income variable by gender
graph box income, over(gender) title ("Income by Gender") ytittle ("Income ($)") // presenting the graph by title
	 
*Shapiro-Wilk test and Skewness/ kurtosis test using Statistically
swilk varname // Shapiro–Wilk test, if p > 0.05 → data normally distributed, 
              // while if p < 0.05 → data likely not normally distributed 
sktest varname // Skewness/Kurtosis test; Skewness between −1 and +1 → roughly symmetric and 
               // Excess kurtosis near 0 → close to normal.

**Reliability and validity test*
alpha m1_phq9a m1_phq9b m1_phq9c m1_phq9d m1_phq9e m1_phq9f ///
 m1_phq9g m1_phq9h m1_phq9i,item std //at baseline
alpha m2_205a_r1 m2_205b_r1 m2_205c_r1 m2_205d_r1 m2_205e_r1 ///
m2_205f_r1 m2_205g_r1 m2_205h_r1 m2_205i_r1 ,item std // Survey two
alpha m2_205a_r2 m2_205b_r2 m2_205c_r2 m2_205d_r2 m2_205e_r2 ///
m2_205f_r2 m2_205g_r2 m2_205h_r2 m2_205i_r2, item std // survey three
alpha m2_205a_r3 m2_205b_r3 m2_205c_r3 m2_205d_r3 m2_205e_r3 ///
 m2_205f_r3 m2_205g_r3 m2_205h_r3 m2_205i_r3, item std // Survey four
*--------------------------------------------------------------
*Proportion of antenatal depression across survey time, abd by trimester level
*--------------------------------------------------------------
*Proportion of antenatl depressive symptoms in each time point
{graph bar depre_cat1 depre_cat2 depre_cat3 depre_cat4 depre_cat5 depre_cat6 depre_cat7, ///
    title("Antenatal depressive symptoms across measurement point") ///
    ytitle("Percentage") ///
    blabel(bar) // in wide data
	
preserve
graph bar (percent) if depre_cat==1 & !missing(depre_cat, time), ///
over(time) title("Percentage of women reporting depressive symptoms across time") ///
ytitle("Percentage") ylabel(0(20)100, grid) blabel(bar, format(%5.2f))  //ading the poortion in the bar 2 decimal
restore

* or
graph bar (percent) if depre_cat==1, over(time) ///
title("Antenatal depressive symptoms across time") ///
ytitle("Percentage") ylabel(0(10)100, nogrid) ///
blabel(bar, format(%5.2f) size(small) color(black)) bar(1, fcolor(navy*0.3) lcolor(black)) ///
bar(2, fcolor(navy*0.5) lcolor(black)) bar(3, fcolor(navy*0.7) lcolor(black)) ///
bar(4, fcolor(navy*0.9) lcolor(black)) graphregion(color(white) lcolor(white)) ///
plotregion(color(white) lcolor(white)) // very important and published graph but x-axis titile name  is difficult to onter

  *OR
graph bar (percent) if depre_cat==1, over(time) ///
title("Antenatal depressive symptoms (Yes) across time", ///
      size(large) margin(t+10)) ///
ytitle("Percentage", size(medlarge) margin(r+8)) ///
ylabel(0(20)100, grid) ///
blabel(bar, format(%5.2f) size(small) color(black)) ///
bar(1, fcolor(navy*0.6) lcolor(black)) ///
graphregion(color(white)) plotregion(margin(large)) // not working due to erro on xtitle option*
   *OR
graph bar (percent) if depre_cat==1, over(time) ///
title("Antenatal depressive symptoms across time", ///
      size(large)  margin(t+10)) ///
ytitle("Percentage", size(medlarge) margin(r+8)) ///
///xtitle("Measurement Time", size(medlarge) margin(t+5)) /// //
ylabel(0(10)100, nogrid) ///
blabel(bar, format(%5.2f) size(small) color(black)) ///
bar(1, fcolor(navy*0.6) lcolor(black)) ///
graphregion(color(white) lcolor(white)) ///
plotregion(color(white) lcolor(white))} // Not used


*-----------------------------------------------------------------------------
*Atenatl depression overtime line graph: across 7 time point
*-----------------------------------------------------------------------------
preserve
keep if !missing(depre_cat, time)
collapse (mean) prop_exp = depre_cat (count) n = depre_cat, by(time)
gen se = sqrt(prop_exp * (1 - prop_exp) / n)
gen ci_low  = prop_exp - 1.96 * se
gen ci_high = prop_exp + 1.96 * se

* Convert to percentage scale
replace prop_exp = prop_exp * 100
replace ci_low  = ci_low  * 100
replace ci_high = ci_high * 100

//replace ci_low = max(ci_low, 0)
//replace ci_high = min(ci_high, 1)
//gen prop_label = string(round(prop_exp*100, 0.1), "%4.1f") //making the exact value by one decimal

gen prop_label = string(round(prop_exp, 0.1), "%4.1f") //making the exact value by one decimal
*Label time points
label define timelab 1 "Baseline" 2 "Month 1" 3 "Month 2" 4 "Month 3" 5 "Month 4" 6 "Month 5" 7 "Month 6"
label values time timelab
twoway ///
    (rarea ci_high ci_low time, color(maroon%25)) ///
    (line prop_exp time, lcolor(red) lwidth(medium)) ///
    (scatter prop_exp time, mcolor(maroon) msymbol(circle) msize(medsmall)) ///
    (scatter prop_exp time, mlabel(prop_label) mlabpos(12) ///
        mlabsize(medium) mlabcolor(black) msymbol(none)), /// //adding the exact value in the line
    ytitle("Antenatal depressive symptoms(%)", size(medlarge) margin(3 3 3 3)) ///
    xtitle("Measurment time", size(medium) margin(3 3 3 3)) ///
    ylabel(0(25)100, labsize(medium) nogrid) ///
    xlabel( 1 "Baseline" 2 "Month 1" 3 "Month 2" 4 "Month 3" 5 "Month 4" 6 "Month 5" 7 "Month 6", labsize(small)nogrid) ///
    legend(off) ///
    title("A. Across measurment time", size(medlarge) margin(0 0 0 0) placement(12)) /// 
    graphregion(color(white)) ///
    plotregion(lstyle(none) margin(5 5 5 5)) ///
	ysize(5.2)
graph save dep_panelB_maroon.gph, replace
restore //best used for presentation

*-----------------------------------------------------------------------------
*Antenatal depressive symtoms at baseline by trimeterlevel:Using scatter 95%CI
*----------------------------------------------------------------------------
preserve

* Keep baseline only
keep if time==1

* Remove missing values
keep if !missing(depre_cat, trimester)

* Collapse to proportion
collapse (mean) prop_dep = depre_cat (count) n = depre_cat, by(trimester)

* Standard error
gen se = sqrt(prop_dep*(1-prop_dep)/n)

* 95% CI
gen ci_low  = prop_dep - 1.96*se
gen ci_high = prop_dep + 1.96*se

* Convert to percentage
replace prop_dep = prop_dep*100
replace ci_low   = ci_low*100
replace ci_high  = ci_high*100

* Prevent CI outside range
replace ci_low  = max(ci_low,0)
replace ci_high = min(ci_high,100)

* Label values
gen prop_label = string(round(prop_dep,0.1),"%4.1f")

* Trimester labels
label define trimlab 1 "1st trimester" 2 "2nd trimester" 3 "3rd trimester", replace
label values trimester trimlab

* Graph: shaded CI + line + scatter
twoway ///
    (rarea ci_high ci_low trimester, color(maroon%25)) /// shaded CI
    (line prop_dep trimester, lcolor(red) lwidth(medium)) /// line
    (scatter prop_dep trimester, mcolor(maroon) msymbol(circle) msize(medium)) /// points
    (scatter prop_dep trimester, mlabel(prop_label) mlabpos(12) ///
        mlabsize(large) mlabcolor(black) msymbol(none)), /// labels
    ylabel(0(25)100, labsize(medium) nogrid) ///
    xlabel(1 "First" 2 "Second" 3 "Thired", labsize(medium)nogrid) ///
    ytitle("", size(medlarge) margin(3 3 3 3)) ///
    xtitle("", size(medlarge) margin(0 0 0 0)) ///
    legend(off) /// 
	title("B. Baseline", size(large) margin(0 0 0 0) placement(12)) ///
    graphregion(color(white)) ///
    plotregion(lstyle(none) margin(8 8 8 8)) ///
	ysize(5.2)
graph export depre_trimester_baselineA.png, width(2000) replace
graph save depre_trimester_baselineA.gph, replace
restore

*------------------------------------------------------------------------------
*Antenatal Depressive Symptoms by Trimester During Follow-up: using scatter 95%CI
*------------------------------------------------------------------------------
preserve

* Remove missing observations
keep if !missing(depre_cat, trimester)

* Collapse to proportion by trimester
collapse (mean) prop_dep = depre_cat (count) n = depre_cat, by(trimester)

* Standard error
gen se = sqrt(prop_dep*(1-prop_dep)/n)

* 95% CI
gen ci_low  = prop_dep - 1.96*se
gen ci_high = prop_dep + 1.96*se

* Convert to percentage
replace prop_dep = prop_dep*100
replace ci_low   = ci_low*100
replace ci_high  = ci_high*100

* Ensure CI limits remain within 0–100
replace ci_low  = max(ci_low,0)
replace ci_high = min(ci_high,100)

* Create labels for points
gen prop_label = string(round(prop_dep,0.1),"%4.1f")

* Label trimester
label define trimlab 1 "1st trimester" 2 "2nd trimester" 3 "3rd trimester", replace
label values trimester trimlab

* Graph: shaded CI + line + scatter
twoway ///
    (rarea ci_high ci_low trimester, color(maroon%25)) /// shaded CI
    (line prop_dep trimester, lcolor(red) lwidth(medium)) ///
    (scatter prop_dep trimester, mcolor(maroon) msymbol(circle) msize(medium)) /// points
    (scatter prop_dep trimester, mlabel(prop_label) mlabpos(12) ///
        mlabsize(large) mlabcolor(black) msymbol(none)), /// labels
    ylabel(0(25)100, labsize(medium) nogrid) ///
    xlabel(1 "First" 2 "Second" 3 "Thired", labsize(medium)nogrid) ///
    ytitle("", size(medlarge) margin(0 0 0 0)) ///
    xtitle("", size(medlarge) margin(0 0 0 0)) ///
    legend(off) ///
	title("C. During follow-up", size(large) margin(0 0 0 0) placement(12)) ///
    graphregion(color(white)) ///
    plotregion(lstyle(none) margin(8 8 8 8)) ///
	ysize(5.2)
graph export depre_trimester_followupB.png, width(2000) replace
graph save depre_trimester_followupB.gph, replace
restore

*----------------------------------------------------------------------------
*Combine the two graphs of the baseline and follow-up AND over trimester period
*----------------------------------------------------------------------------
graph combine depre_trimester_baselineA.gph depre_trimester_followupB.gph, ///
    col(2) ///
    imargin(3 3 3 3) ///
    ycommon ///
    graphregion(color(white)) ///
      b1title("Trimester level", size(medlarge) margin(2 2 2 2)) // for making shared x-title
	  //title("Antenatal Depressive Symptoms over Trimester at Baseline and Follow-up", size(medlarge)) ///
graph export deper_baseline_followup.png, width(2000) replace
graph save combined_depre_baseline_followup.gph, replace

*-------------------------------------------------------------------
*Combine the three graph: Antenatal depression over time, baseline and follow-up bt terimester level
*-------------------------------------------------------------------
graph combine dep_panelB_maroon.gph combined_depre_baseline_followup.gph, ///
    col(2) ///
    imargin(3 3 3 3) ///
    ycommon ///
    graphregion(color(white)) ///
      //b1title("Trimester level", size(medium) margin(2 2 2 2)) // for making shared x-title
	  //title("Antenatal Depressive Symptoms over Trimester at Baseline and Follow-up", size(medlarge)) ///
graph export combined_depre_month_baseline_followup.png, width(2000) replace
graph save combined_depre_month_baseline_followup.gph, replace

*---------------------------------------------------------------------
*Antenatl depression at baseline and follow-up over women trmieter level: Using scatter plot 95%CI
*-----------------------------------------------------------------------------
*Antenatal depressive symtoms at baseline
*----------------------------------------------------------------------------
preserve

* Keep baseline only
keep if time==1

* Remove missing values
keep if !missing(depre_cat, trimester)

* Collapse to proportion
collapse (mean) prop_dep = depre_cat (count) n = depre_cat, by(trimester)

* Standard error
gen se = sqrt(prop_dep*(1-prop_dep)/n)

* 95% CI
gen ci_low  = prop_dep - 1.96*se
gen ci_high = prop_dep + 1.96*se

* Convert to percentage
replace prop_dep = prop_dep*100
replace ci_low   = ci_low*100
replace ci_high  = ci_high*100

* Prevent CI outside range
replace ci_low  = max(ci_low,0)
replace ci_high = min(ci_high,100)

* Label values
gen prop_label = string(round(prop_dep,0.1),"%4.1f")

* Trimester labels
label define trimlab 1 "1st trimester" 2 "2nd trimester" 3 "3rd trimester", replace
label values trimester trimlab

* Graph: shaded CI + line + scatter
twoway ///
    (rarea ci_high ci_low trimester, color(maroon%25)) /// shaded CI
    (line prop_dep trimester, lcolor(red) lwidth(medium)) /// line
    (scatter prop_dep trimester, mcolor(maroon) msymbol(circle) msize(medium)) /// points
    (scatter prop_dep trimester, mlabel(prop_label) mlabpos(12) ///
        mlabsize(medlarge) mlabcolor(black) msymbol(none)), /// labels
    ylabel(0(25)100, labsize(medlarge) nogrid) ///
    xlabel(1 "First" 2 "Second" 3 "Thired", labsize(medium)nogrid) ///
    ytitle("Antenatal depressive symptoms (%)", size(large) margin(3 3 3 3)) ///
    xtitle("", size(medlarge) margin(0 0 0 0)) ///
    legend(off) /// 
	title("A. Baseline", size(large) margin(0 0 0 0) placement(12)) ///
    graphregion(color(white)) ///
    plotregion(lstyle(none) margin(8 8 8 8)) ///
	ysize(4.5)
graph export depre_trimester_baselineA.tif, as(tif) name(Graph) width(2000) replace
graph save depre_trimester_baselineA.gph, replace
restore

*------------------------------------------------------------------------------
*Antenatal Depressive Symptoms by Trimester During Follow-up: using scatter 95%CI
*------------------------------------------------------------------------------
preserve

* Remove missing observations
keep if !missing(depre_cat, trimester)

* Collapse to proportion by trimester
collapse (mean) prop_dep = depre_cat (count) n = depre_cat, by(trimester)

* Standard error
gen se = sqrt(prop_dep*(1-prop_dep)/n)

* 95% CI
gen ci_low  = prop_dep - 1.96*se
gen ci_high = prop_dep + 1.96*se

* Convert to percentage
replace prop_dep = prop_dep*100
replace ci_low   = ci_low*100
replace ci_high  = ci_high*100

* Ensure CI limits remain within 0–100
replace ci_low  = max(ci_low,0)
replace ci_high = min(ci_high,100)

* Create labels for points
gen prop_label = string(round(prop_dep,0.1),"%4.1f")

* Label trimester
label define trimlab 1 "1st trimester" 2 "2nd trimester" 3 "3rd trimester", replace
label values trimester trimlab

* Graph: shaded CI + line + scatter
twoway ///
    (rarea ci_high ci_low trimester, color(maroon%25)) /// shaded CI
    (line prop_dep trimester, lcolor(red) lwidth(medium)) ///
    (scatter prop_dep trimester, mcolor(maroon) msymbol(circle) msize(medium)) /// points
    (scatter prop_dep trimester, mlabel(prop_label) mlabpos(12) ///
        mlabsize(medlarge) mlabcolor(black) msymbol(none)), /// labels
    ylabel(0(25)100, labsize(medlarge) nogrid) ///
    xlabel(1 "First" 2 "Second" 3 "Thired", labsize(medium)nogrid) ///
    ytitle("", size(medlarge) margin(0 0 0 0)) ///
    xtitle("", size(medlarge) margin(0 0 0 0)) ///
    legend(off) ///
	title("B. During follow-up", size(large) margin(0 0 0 0) placement(12)) ///
    graphregion(color(white)) ///
    plotregion(lstyle(none) margin(8 8 8 8)) ///
	ysize(4.5)
graph export depre_trimester_followupB.tif, as(tif) name(Graph) width(2000) replace
graph save depre_trimester_followupB.gph, replace
restore

*----------------------------------------------------------------------------
*Combine the baseline and follow-up of AND over trimester period
*----------------------------------------------------------------------------
graph combine depre_trimester_baselineA.gph depre_trimester_followupB.gph, ///
    col(2) ///
    imargin(3 3 3 3) ///
    ycommon ///
    graphregion(color(white)) ///
      b1title("Trimester level", size(medsmall) margin(2 2 2 2)) /// // for making shared x-title
	 ysize(4.5) 
	  //title("Antenatal Depressive Symptoms over Trimester at Baseline and Follow-up", size(medlarge)) ///
graph export deper_baseline_followup.tif, as(tif) name(Graph) width(2000) replace //using TIF
graph save combined_depre_baseline_follow.gph, replace // used in the maunscript


******************************************************************************
   "For exporting graph"
***************************************************************************** 
graph export "depression_yes_time.png", as(png) name(Graph) width(2000) replace //Using png
graph export "depression_yes_time.tif", as(tif) name(Graph) width(4000) replace //using TIF
graph export "depression_yes_time.eps", as(eps) name(Graph) replace // usi g EPS
graph export "depression_yes_time.pdf", as(pdf) name(Graph) replace //Using pdf

********************************************************************************

*By trimester category
graph bar depre_cat1 depre_cat2 depre_cat3 depre_cat4 depre_cat5 depre_cat6 depre_cat7, ///
    over(m1_trimester)

graph bar percent, over(service, gap(5)) blabel(bar, format(%9.1f)) ///
    title("Maternal Health Service Coverage") ///
    ytitle("Percentage") ///
    bar(1, color(navy)) bar(2, color(blue*0.5)) bar(3, color(ltblue))
	
*Depression at survey one
tab	depre_cat1 

*Depression at survey two/
tab depre_cat2
	  *Then
*To determine the total number of pregnant women who experienced depressive symptoms both times one and two*
keep if depre_cat2==1 //to filter only those pregnant women who experienced depressive symptoms at survey two
	  *Then 
tab depre_cat1 // to know off those who experienced ANDS at survey2,
                   // who many of them experienced survey one also*
	
{egen pers_depre_cat_s1s2=rowtotal(depre_cat1 depre_cat2), mi
lab var pers_depre_cat_s1s2 "Persistent experience of deprrssive symptoms considering two time points"} //this command is done without keeping the variable
	
*Depression at survey three/time three*
tab depre_cat3
	   *Then
*To determine the total number of pregnant women who experienced depressive symptoms both times: one and two*
keep if depre_cat3==1 // to filter only those pregnant women who experienced depressive symptoms at survey three
	 *Then 
tab depre_cat2 // to know off those who experienced ANDS at survey3, who many of them experienced survey two also*
tab depre_cat2 // to know off those who experienced ANDS at survey3, who many of them experienced survey one also*
	
{egen pers_depre_cat_s1s3=rowtotal (depre_cat1 depre_cat2 depre_cat3), mi
lab var pers_depre_cat_s1s3 "Persistent experience of deprrssive symptoms considering three time point"} ////this command is done without keeping the variable
	
*Depression at survey four/time four*
tab depre_cat4
	  *Then
*To determine the total number of pregnant women who experienced depressive symptoms both times: one, two, and three*
keep if depre_cat4==1 // to filter only those pregnant women who experienced depressive symptoms at survey three
	  *Then 
tab depre_cat3 // to know off those who experienced ANDS at survey4, who many of them experienced survey three*
tab depre_cat2 // to know off those who experienced ANDS at survey4, who many of them experienced survey two*
tab depre_cat1 // to know off those who experienced ANDS at survey4, who many of them experienced survey one*

*Considering seven time points
egen pers_depre_cat_s1s7= rowtotal(depre_cat1 depre_cat2 depre_cat3 ///
depre_cat4 depre_cat5 depre_cat6 depre_cat7), mi 
lab var pers_depre_cat_s1s7 "Persistent experience of depressive symptoms considering seven time points" 
ta pers_depre_cat_s1s7

*Women not experienced antnatal depressive symptoms at all times 
recode pers_depre_cat_s1s7 0=1 1/max=0, gen(depre_not_alltime)
lab def depre_not_alltime 1 "Yes" 0 "No"
lab val depre_not_alltime depre_not_alltime
ta depre_not_alltime

*Women experienced of antenatal depressive symptoms at least once
recode pers_depre_cat_s1s7 0=0 1/max=1, gen(depre_atleast_once)
lab def depre_atleast_once 1 "Yes" 0 "No"
lab val depre_atleast_once depre_atleast_once
ta depre_atleast_once

*Women experienced antenatal depressive symptoms at least twice
recode pers_depre_cat_s1s7 0/1=0 2/max=1, gen(depre_atleast_twice)
lab def depre_atleast_twice 1 "Yes" 0 "No"
lab val depre_atleast_twice depre_atleast_twice
ta depre_atleast_twice

*Women experienced antenatal depressive symptoms at three times
recode pers_depre_cat_s1s7 0/2=0 3/max=1, gen(depre_atleast_three)
lab def depre_atleast_three 1 "Yes" 0 "No"
lab val depre_atleast_three depre_atleast_three
ta depre_atleast_three

*Women experience antenatal depressive symptoms at four times
recode pers_depre_cat_s1s7 0/3=0 4/max=1, gen(depre_atleast_four)
lab def depre_atleast_four 1 "Yes" 0 "No"
lab val depre_atleast_four depre_atleast_four
ta depre_atleast_four

*Women experienced antenatl depressive symptoms at five times
recode pers_depre_cat_s1s7 0/4=0 5/max=1, gen(depre_atleast_five)
lab def depre_atleast_five 1 "Yes" 0 "No"
lab val depre_atleast_five depre_atleast_five
ta depre_atleast_five

*Women experienced antenatl depressive symptoms at six times
recode pers_depre_cat_s1s7 0/5=0 6/max=1, gen(depre_atleast_six)
lab def depre_atleast_six 1 "Yes" 0 "No"
lab val depre_atleast_six depre_atleast_six
ta depre_atleast_six

*Women experienced antenatl depressive symptoms at seven times
recode pers_depre_cat_s1s7 0/6=0 7/max=1, gen(depre_atleast_seven)
lab def depre_atleast_seven 1 "Yes" 0 "No"
lab val depre_atleast_seven depre_atleast_seven
ta depre_atleast_seven

order pers_depre_cat_s1s7 depre_not_alltime depre_atleast_once ///
depre_atleast_twice depre_atleast_three depre_atleast_four ///
depre_atleast_five depre_atleast_six depre_atleast_seven, after (ancdepre_cat8)
 ************************************************************************* 
 "Trajectory of antenatal depressive symptoms"
*******************************************************************************
*Install the package of the trajectory model
[net from https://www.andrew.cmu.edu/user/bjones/traj
net install traj, replace] // run at once both

*Create the time variable for each survey time (surveyti) in wide format, 
    //considering the ni survey times
m1_date m2_date_r1 m2_date_r2 m2_date_r3 m2_date_r4 m2_date_r5
*Survey during pregnancy (8 time points but only considered four survey point for 1st objective)
recode m1_date    (0/max=1) (.=1), gen(survey1)
recode m2_date_r1 (0/max=2) (.=2), gen(survey2)
recode m2_date_r2 (0/max=3) (.=3), gen(survey3)
recode m2_date_r3 (0/max=4) (.=4), gen(survey4)
recode m2_date_r4 (0/max=5) (.=5), gen(survey5)
recode m2_date_r5 (0/max=6) (.=6), gen(survey6)
recode m2_date_r6 (0/max=7) (.=7), gen(survey7)
recode m2_date_r7 (0/max=7) (.=7), gen(survey8)

*Survey at after delivery(only three time points)
recode m3_date_p2 (0/max=1) (.=1), gen(pncsurvey1)
recode m4_102     (0/max=2) (.=2), gen(pncsurvey2)
recode m5_date    (0/max=3) (.=3), gen(pncsurvey3)
 
*Persistent experience of antenatal depressive symptoms
recode pers_dep_5cut_s1s4 (0=0) (1/4=1), gen(experienced_1times) //experienced one times only
recode pers_dep_5cut_s1s4 (0/1=0) (2/4=1), gen(experienced_2times) //experienced two times
recode pers_dep_5cut_s1s4 (0/2=0) (3/4=1), gen(experienced_3times) //experienced three times
recode pers_dep_5cut_s1s4 (0/3=0) (4=1), gen(experienced_4times) //experienced all times


*Group-based-trajectory modeling (GBTM)
traj, var(outcomet1-t5) indep(survey1-5) model(cnorm) order(2 2) min(0) max(27)

**two group order**
traj, var(depressions1 depressions2 depressions3 depressions4 depressions5 depressions6 ) ///
indep( survey1 survey2 survey3 survey4 survey5 survey6) model(cnorm) order(2 2 2) min(0) max(27) // for contious outcomes
 
traj, var(depre_cat1 depre_cat2 depre_cat3 depre_cat4 depre_cat5 depre_cat6) ///
     indep(survey1 survey2 survey3 survey4 survey5 survey6) ///
     model(logit) order(2 2 2) min(0) max(1) // catagorical

	 *Plot trajectory
*To visualize trajectories (average predicted probability or mean for each group)	 
trajplot, ci

trajplot, xtitle("Survey time") ytitle("Predicted probability") ///
          title("Trajectories of depressive symptoms") ///
          legend(order(1 "Low-stable" 2 "Moderate-increasing" 3 "High-persistent"))
*for checking the trajectory of six time point completed measurement**
*first keep only those who completed the six measure using this (221 women)
keep if !missing(depre_cat1, depre_cat2, depre_cat3, depre_cat4, depre_cat5, depre_cat6)

*then excute the trajectory		  
 traj, var(depre_cat1 depre_cat2 depre_cat3 depre_cat4 depre_cat5 depre_cat6) ///
     indep(survey1 survey2 survey3 survey4 survey5 survey6) ///
     model(logit) order(2 2 2) min(0) max(1) // catagorical
	 
traj, var(depressions1 depressions2 depressions3 depressions4 depressions5 depressions6 depressions7 ) ///
 indep( survey1 survey2 survey3 survey4 survey5 survey6 survey7)/// 
 model(cnorm) order(2 2 2) min(0) max(27) // for contious variables
 
   // Check the AIC, BIC, aBIC,  and group size in each class (>5% is an acceptable group size), 
   // and entropy (>0.7 is good classification)

*Evaluate model quality
*After fitting, check:
  *Group size: at least 5% of total sample per class
  *Average posterior probability (AvePP): should be ≥ 0.7 for each group
  *Odds of correct classification (OCC): should be > 5

*Plot Trajectories
 trajplot, ci xtitle ("Survey time") ytitle ("Depression score")
   *or
 {trajplot, ci(rarea) lcolor (red green blue) fcolor(red*0.2 green*0.2 blue*0.2) ///
 xlabel(1 2 3 4) ylabel(0(5)27) ytitle("PHQ-9 score") xtitle("Survey time") ///
 title("Trajectory of antenatal depressive symptoms") ///
 legend(order(3 "Persistent class" 2 "Moderate decreasing class" 1 "Improving class") ///
 ring(0) pos(1) col(1)) yline(10, lpattern(dash) lcolor(black))} 
 *or
traj-plot, ci(rarea) lcolor(red blue black) fcolor(red*0.15 blue*0.15 black*0.8) ///
 xlabel(1 2 3 4) ylabel(0(5)27) ytitle("PHQ-9 score", size(medium)) ///
 xtitle("Survey time", size(medium)) title("Trajectory of antenatal depressive symptoms") ///
 legend(order(3 "Persistent (Class 3: n=82; 11.3%)" 2 "Moderate decreasing (Class 2: n=385; 49.2%)" ///
 1 "No depressive (Class 1: n=287; 39.5%)") ring(0) pos(2) col(1) size(small)) /// 
 xline(2, lpattern(dash) lcolor(black))} // Not working, will check the command again

*Check group sizes
 tab _traj_Group // showing the % of participants in each group, small groups (<5%) may not be reliable.

*Check Posterior Probabilities of each group
 summ _traj_ProbG1 _traj_ProbG2 _traj_ProbG2 // for two group

*Determine the average posterior probability (APP) of each group
 summ _traj_ProbG1 if _traj_Group==1 // group one
 summ _traj_ProbG2 if _traj_Group==2
 // group two
  summ _traj_ProbG3 if _traj_Group==3
*Interpretation:
 //Higher APP (>0.7) → participants in that group were classified with high certainty/acceptable classification of group.
  
**Three group order**
 traj, var(depressions1 depressions2 depressions3 depressions4 depressions5 depressions6 depressions7 ) ///
 indep( survey1 survey2 survey3 survey4 survey5 survey6 survey7 ) model(cnorm) order(2 2) min(0) max(27) // 
 
 proportion _traj_Group // proportion of trajectory group with CI
 
   // Check the AIC, BIC, aBIC, group size in each class (>5% is an acceptable group size/classification), and entropy (>0.7 is good classification)
*Plot Trajectories
 trajplot, ci xtitle ("Survey time") ytitle ("Depression score") // for plotting the trajectory graph
*Or
 {trajplot, ci(rarea) lcolor (red green blue) fcolor (red*0.2 green*0.2 blue*0.2) ///
 xlabel(0 4 8 12) ylabel(0(5)30) ytitle("PHQ9 score") xtitle("Survey") ///
 title("depression trajectort") legend(order(3 "Persistent class" ///
 2 "Moderate decreasing class" 1 "Improving class") ring(0) pos(1) col(1)) ///
 yline(10, lpattern(dash) lcolor(black))} //not working, will be cheked the command again

*Check Group sizes
 tab _traj_Group // showing the % of participants in each group, small groups (<5%) may not be reliable.

*Check Posterior Probabilities of each group
 summ _traj_ProbG1 _traj_ProbG2 _traj_ProbG3 // three group

*Determine the APP of each group
 summ _traj_ProbG1 if _traj_Group==1 // group one
 summ _traj_ProbG2 if _traj_Group==2 // group two
 summ _traj_ProbG3 if _traj_Group==3 // group three
*Interpretation:
 //Higher APP (>0.7) → participants in that group were classified with high certainty/acceptable classification of group.
* NB: all the remaining group 4 and five are similar

*After selecting the trajectory model, we can describe the trajectory of depressive symptoms by participants' profiles
 tab _traj_Group sex, chi2 // by sex
 tab _traj_Group age, chi2 //
 tab _traj_Group trimester1, chi2 //trimester
 
**for area shading
traj, var (y1 y2 y3 y4 y5) indep(1 2 3 4 5) model(cnorm) order(2 2 2) ngroups(3) min(0)
trajplot, ci


*GBTM when the data is wide format
 // Considering five groups of trajectory modeling
 // Create five groups for each individual
{gen group =.
replace group = 1 in 1/145
replace group = 2 in 146/291
replace group = 3 in 292/437
replace group = 4 in 438/583
replace group = 5 in 584/727

* Generate outcome y(depression) for each trajectory group
gen depression =.
replace depression = 2 + 0.4*time + rnormal(0,0.25) if group==1
replace depression = 3 + 0.2*time + rnormal(0,0.25) if group==2
replace depression = 4 - 0.3*time + rnormal(0,0.25) if group==3
replace depression = 3.5 + 0.0*time + rnormal(0,0.25) if group==4
replace depression = 2.5 - 0.1*time + rnormal(0,0.25) if group==5}

******************************************************************************
"Determinant of antenatal depressive symptoms" 
   Done considering six times
  Using GEE_Logit link function
****************************************************************************
  *Open the six time point long and wide dataset
  
logistic depre_cat1 m1_anemia_px_edit, vce(robust)
	  
**For GEE analysis*
 *Declare the panel data
xtset idn
xtset idn time
*Determinant of antenatal depressive symptoms
*For running correlation matrix
xtcorr
*Working correlation command for independent (corr(ind)); 
*for Exchangeable (coor(exc)), for unstructured (corr(un)), 
*for Autoregressive (corr(ar1)), corr(stationary 1), corr(nonstationary 1)

*First select the right working correlation structure
xtgee depre_cat, family(binomial 1) link(logit) corr(un) vce(robust) eform
xtcorr // look the correlation
  // The correlation structure after fitting the model using unstructured working correlation
  // the correlation structure showed an exchangeable because the correlation b/n measurement nearly similar 
  // rather than decreasing as power of measurement deference (AR1) or the lower correlation not close to zero (Independent)
  // additional the QiC value is lower in exchangeable correlation as compared with other
  // following this every analysis is based on exchangeable working correlation 
  // As a result, for every analysis was done considering the exchangeable, and
  // the final model was selected by using  QICU (the lower value is the best fit final model)

**Bi-variable analysis: considering an autoregressive WCR
*Educational status (Reference=3/More educated)
xtgee depre_cat ib3.m1_edulevel_3cat, family(binomial 1) ///
link(logit) corr(exc) vce(robust) eform
   
*Marital status (reference: 1/married)
xtgee depre_cat ib0.m1_marital, family(binomial 1) ///
link(logit) corr(exc) vce(robust) eform
	
*Health literacy of the respondent (Ref: first "poor") m1_hliteracy_2cat
xtgee depre_cat ib2.m1_health_literacy, family(binomial 1) ///
link(logit) corr(exc) vce(robust) eform // not sign/ p-value 0.145
 
*Age of the respondent (reference: last ">=35 years")
xtgee depre_cat ib3.m1_age_cat1, family(binomial 1) ///
link(logit) corr(exc) vce(robust) eform	// notsign/ p-value 0.625
	 
*Occupation (ref, last option "others")
xtgee depre_cat ib4.m1_occupation, family(binomial 1) link(logit) corr(exc) vce(robust) eform

*Women's economic status using the Wealth index (ref is the last "poorest")
xtgee depre_cat ib3. tertile, family(binomial 1) link(logit) corr(exc) vce(robust) eform
   
*Religion of the respondent (reference: first/Christian 
xtgee depre_cat ib3.m1_religion , family(binomial 1) link(logit) corr(exc) vce(robust) eform // non significant/0.054

*Family support (m1_familysupport; reference: 1/first, less than one family)
xtgee depre_cat ib2.m1_familysupport , family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform	// notsign/ 0.15  

*Women trimester level (ref, 3/third trimester)
xtgee depre_cat ib3.m1_trimester, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform // not sign , p-value=0.298 /basline

xtgee depre_cat ib3.trimesters, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform // not sign/p-value=0.145/repeated/considered

*Pregnancy intention (ref, intended pregnancy (1/yes)
xtgee depre_cat ib1.preg_intent , family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform

*Intimate partner violence (Ref, 0/No)
xtgee depre_cat ib0.m1_ipv_px_cat, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform
 
*Substance use (Ref: first "0/no")
xtgee depre_cat ib0.subuse_cat, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform // not sign/ p-value 0.388

*History of chronic illness (Ref: first "0/No)"
xtgee depre_cat ib0.m1_Hx_chronic_illness, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform //not sign p-value 0.296

*Currently diagnosed chronic illness
xtgee depre_cat ib0.m1_cur_chronic_illness, family(binomial 1) /// 
 link(logit) corr(exc) vce(robust) eform //not sign p-value 0.907

*Chronic illness (both history and current diagnosis ref: First "O/no")
xtgee depre_cat ib0.m1_chronic_illness, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform //not sign p-value 0.377
 
*Self-reported health status (ref, 1/good excellent)
xtgee depre_cat ib1.healthstatus, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform // repeated measure
 
*Women's health health-related quality of life/WHRQoL (Ref:3/poor)
xtgee depre_cat ib3.anc_hrqol_cat, family(binomial 1) link(logit) corr(exc) vce(robust) eform 

*Women's health health-related quality of life/WHRQoL (Ref:3/poor)
xtgee depre_cat ib1.anc_hrqol_2cat, family(binomial 1) link(logit) corr(exc) vce(robust) eform
*Maternal anemia (Ref:0/No anemic)
xtgee depre_cat ib0.m1_anemia_px_edit_02, family(binomial 1) link(logit) corr(exc) vce(robust) eform

xtgee depre_cat ib0.m1_anemia_px, family(binomial 1) link(logit) corr(exc) vce(robust) eform

*Maternal malnutrition (Ref: 0/not malnourished)
xtgee depre_cat ib0.m1_malnutrition, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform //not sign p-value 0.87

*Maternal undernutrition (Ref: 0/No)
xtgee depre_cat ib0.m1_undernutrition, family(binomial 1) ///
 link(logit) corr(exc) vce(robust) eform //not sign p-value 0.391
	
*Experienced common health problems (Ref, 0/No)
xtgee depre_cat ib0.m1_comhealth_problem, family (binomial 1) ///
 link(logit) corr(exc) vce(robust) eform // p-value=0.000 but a couldn't considered

*Experienced at least one obstetric problem in the current pregnancy (Ref: 0/No)
xtgee depre_cat ib0.obstproblem_cat, family(binomial 1) link(logit) corr(exc) vce(robust) eform

*Experienced at least one pregnancy danger sign (Ref, 0/No)
xtgee depre_cat ib0.pxdanger_cat, family(binomial 1) link(logit) corr(exc) vce(robust) eform
	
**Multivariate analysis (occupation and common health problem variables were considered as not significant)
xtset idn time

**For checking VIF**
*Run
regress depre_cat ib3.m1_edulevel_3cat ib0.m1_marital ib4.m1_occupation /// 
 ib3.tertile ib1.m1_religion ib2.m1_familysupport ib3.trimesters /// 
 ib1.preg_intent ib0.m1_ipv_px_cat ib1.healthstatus ///
 ib0.anc_ihrol_2cat ib0.m1_anemia_px_edit_02 ib0.m1_comhealth_problem ///
 ib0.obstproblem_cat ib0.pxdanger_cat, vce(robust) 
 preditc
 // Then run "vif" 
vif 

*OR
regress depressions ib3.m1_edulevel_3cat ib0.m1_marital ib4.m1_occupation /// 
 ib3.tertile ib1.m1_religion ib2.m1_familysupport ib3.trimesters /// 
 ib1.preg_intent ib0.m1_ipv_px_cat ib1.healthstatus ///
 ib0.anc_ihrol_2cat ib0.m1_anemia_px_edit_02 ib0.m1_comhealth_problem ///
 ib0.obstproblem_cat ib0.pxdanger_cat, vce(robust)  eform 
 // then run "vif"
vif
 
**Outlier identifcation and management* 
*Using standardized residual
predict resid, r  // for identifying outlier
summ resid, detail  // summar of outliers
scalar sd_resid = r(sd)
drop if abs(resid) > 2*sd(resid) // for dropping outliers with >2 of studentized residual value
                                 // Compare coefficients two models (before and after excluding outliers).
								 // "abs(resid)" indicates the absolute value of the residual
								 // "sd" indicates standard deviation
								 // "resid" indicates residual
*Then, after doing these three command consequtively number of the outlier identifred by diong this
  summ resid
  
scalar sd_resid = r(sd) // instead of dropping good to generte outlier then inspect them
gen outlier = abs(resid) > 2*sd_resid 
tab outlier								 
								 
*Using Z-score*
summ resid, meanonly
gen zscore = (resid - r(mean)) / r(sd)
list if abs(zscore) > 3  // Any observation with |zscore| > 3 is an extreme outlier.
							 
*Using the interqurtile range (IQR)	
summ resid, detail
gen IQR = r(p75) - r(p25)
gen lower = r(p25) - 1.5*IQR
gen upper = r(p75) + 1.5*IQR
list if resid < lower | resid > upper //Those beyond this range are potential outliers.

*Leverage and cook distance
Influential points: high Cook's D (> 4/n), Leverage, or DFBETAs	
				 
*Check if the model is correctly specified (whether we omitted important variables from the model, and the model you consider is misspecified)
estat ovtest // Ramsey RESET test for omitted variable bias, to check whether the important variables are omitted or not
             // if the Prob > F is grater then 0.05 the model is not ommitted important variables, 
		     // which shows our model specification looks acceptable (reasonably well-specified).
				
linktest // Link test for model misspecification
        // the link test checks for model specification errors as well, 
	   // it tests whether the model's predicted values adequately capture the relationship between predictors and the outcome.
            // If p-values "_hat" (the model's predicted value) is <0.05, showing the model predicts the outcome and,
			// the model specification is acceptable.
			// if p-value of "_hatsq"(which is the square of the predicted value) is >0.05 indicating the model is well specified.
**As aresuly "Ramsey RESET test" and "link test" and residual test should done for evry model 
		 
**Model building*
 // Family support variable excluded from analysis, no effect in the final mode at all
 
*Model one: including variables with p-value<0.05 (8 variables): usining sit time points
xtgee depre_cat ib0.m1_marital ib3.tertile  ib1.preg_intent ///
ib0.m1_ipv_px_cat ib1.healthstatus ib1.anc_hrqol_2cat  ///
ib0. obstproblem_cat ib0.pxdanger_cat, ///
family(binomial 1) link(logit) corr(exc) vce(robust) eform // then run "QICU"
     // looking again by adding these ib0.m1_comhealth_problem ib2.m1_familysupport ib4.m1_occupation 

*Checking the QICU of the model
qic depre_cat m1_marital tertile preg_intent m1_ipv_px_cat ///
healthstatus anc_hrqol_2cat obstproblem_cat pxdanger_cat, ///
 family(binomial 1) link(logit) corr(exc) //qicu:7time opoint(4223) vs 6 time point (4132) vs 5 time point (3877) vs 4 time point(2908)

*Model two:includinhg variables with p-value<0.05 & those with marginal significant (11 variables): six time points
 // adding educational status religion and trimesters (p-value bn 0.05 to 0.25)
xtgee depre_cat ib3.m1_edulevel_3cat ib0.m1_marital ib3.tertile ib3.m1_religion /// 
ib3.trimesters ib1.preg_intent ib0.m1_ipv_px_cat ib1.healthstatus ///
ib1.anc_hrqol_2cat ib0.pxdanger_cat ib0. obstproblem_cat, ///
family(binomial 1) link(logit) corr(exc) vce(robust) eform // used in the final model
 
*Checking the QICU of the model 
qic depre_cat m1_edulevel_3cat m1_marital tertile m1_religion trimesters ///
preg_intent m1_ipv_px_cat healthstatus anc_hrqol_2cat pxdanger_cat ///
obstproblem_cat, family(binomial 1) link(logit) corr(exc) // adding education and trimesters; qic:4169.9 vs 4075 vs 3813 vs 3046


//Note: After doing this, run the model using 11 variables with exchangable WCs 
// to select the working correlation and the lower QiC is the best working structure,  
// and the lower QiCu is the best fit final model
**************************************************************************************
*_-----------------------------------------------------------------------------
   "Sensitiveity analysis"
*------------------------------------------------------------------------------
*Model xx: considering (6 completed observations of depression measure: n=221/1326 or 216/1284)
  // Religion & pregnancy intention were excluded (their p-value>0.25)
  
xtgee depre_cat ib3.m1_edulevel_3cat ib0.m1_marital ib3.tertile ///
ib3.trimesters ib0.m1_ipv_px_cat ib1.healthstatus ///
ib1.anc_hrqol_2cat ib0.pxdanger_cat ib0. obstproblem_cat, ///
family(binomial 1) link(logit) corr(exc) vce(robust) eform

qic depre_cat m1_edulevel_3cat m1_marital tertile trimesters ///
m1_ipv_px_cat healthstatus anc_hrqol_2cat pxdanger_cat ///
obstproblem_cat, family(binomial 1) link(logit) corr(exc) // qicu=1428.445, trace=15.795 

*Model xx: considering (only who evaluated at 6 measurement points: n=240/1440 or 240/1409)
  // Educational status, wealth index & religion & were excluded (their p-value>0.25)
  
xtgee depre_cat ib0.m1_marital ib3.trimesters ib1.preg_intent ///
ib0.m1_ipv_px_cat ib1.healthstatus ib1.anc_hrqol_2cat ib0.pxdanger_cat ///
ib0.obstproblem_cat, family(binomial 1) link(logit) corr(exc) vce(robust) eform

qic depre_cat m1_marital trimesters preg_intent m1_ipv_px_cat healthstatus ///
anc_hrqol_2cat pxdanger_cat obstproblem_cat, family(binomial 1) link(logit) corr(exc) // qicu=1571.893, tracer:13.289

*Model xx: considering those who completed all seven  depression measurment points (=56/392 or 55/380)
  // four variable including educational status, religion & pregnancy intention with p-value>0.25) and marital status (all are married) were excluded
  
xtgee depre_cat ib3.tertile ib3.trimesters ib0.m1_ipv_px_cat ///
ib1.healthstatus ib1.anc_hrqol_2cat ib0.pxdanger_cat ///
ib0.obstproblem_cat,family(binomial 1) link(logit) corr(exc) vce(robust) eform

qic depre_cat tertile trimesters m1_ipv_px_cat healthstatus anc_hrqol_2cat ///
pxdanger_cat obstproblem_cat, family(binomial 1) link(logit) corr(exc) 
   // qicu= 422.113, tracer= 13.015

*Model xx: considering (only who evaluated at seven measurement points: n=85/595 or 84/547)
  // religion & pregnancy intention were excluded (their p-value>0.25)
  
xtgee depre_cat ib3.m1_edulevel_3cat ib0.m1_marital ib3.tertile /// 
ib3.trimesters ib0.m1_ipv_px_cat ib1.healthstatus ib1.anc_hrqol_2cat ///
ib0.pxdanger_cat ib0. obstproblem_cat, ///
family(binomial 1) link(logit) corr(exc) vce(robust) eform

qic depre_cat m1_edulevel_3cat m1_marital tertile trimesters m1_ipv_px_cat///
 healthstatus anc_hrqol_2cat pxdanger_cat obstproblem_cat, ///
 family(binomial 1) link(logit) corr(exc) // qicu= 619.689, tracer=19.153

*-------------------------------------------------------------------------------
           *Attrition bias analysis"
 *Completer vs drop out analyses, ' Chi-square, logistic regression, and sensitivity analysis"
                *Using 7-time point dataset*
*-------------------------------------------------------------------------------
***Open the seven-point long dataset
preserve
keep if !missing(depre_cat, time)
ta depre_cat
ta time
restore

*Count the number of observations per participant
keep if !missing(depre_cat, time)

**Estimate Dropout Percentage
*First create completer variable
bysort idn: gen n_rounds= _N 
gen completer = (n_rounds == 7)

lab def completer 0 "Dropped out" 1 "Completed all 7"
lab val completer completer
ta completer

recode completer 0=1 1=0, gen(dropout)
lab def dropout 1 "Dropped out" 0 "Completed all 7"
lab values dropout dropout
ta dropout

*Now estimate dropout rate (baseline sample):
preserve
keep if time==1
tab completer 
restore

*Baseline comparison table
*Checking the presence significant difference between completers and dropouts using baseline characteristic variables and other covariates
*For binary baseline outcome:
preserve
keep if time==1 // 
tab depre_cat completer, row chi2 // p=0.007
ta m1_age_cat1 completer, row chi2 // no significant difference (p=0.138)
ta m1_marital completer, chi2 //p=0.23
ta m1_residence completer, chi2 //p=0.489
ta m1_edulevel_3cat completer, row chi2 //p=0.985
ta tertile completer, row chi2 // 0.093
ta m1_religion completer, row chi2 // p=0.642
ta m1_occupation completer, row chi2 // p=0.311
ta m1_residence completer, row ch //0.489
ta m1_health_literacy completer, chi2 //p=0.517
ta healthstatuss1_2cat completer, row chi2 //0.174
ta m1_malnutrition completer, row chi2 // p=0.292
ta m1_anemia_px completer, row chi2 // p=0.448
ta m1_ipv_px_cat completer, row chi2 // p=0.795
ta m1_risk_health_px_cat completer, row chi2 // p=0.241
ta subuse_cat completer, chi2 // p=0.659
ta m1_trimester completer, chi2 // 0.000
ta preg_intent completer, row chi2 //0.03
ta anc_hrqol_2cat completer, row chi2 //p=0.008
ta anc1px_dangersign completer, row chi2 // p=0.033
ta obstproblem_cat completer, row chi2 // p=0.173
restore

*Optional: Standardized Differences (Stronger) keep time==1
preserve
keep if time==1
//ssc install stddiff
stddiff depre_cat m1_age_cat1 m1_marital m1_residence m1_edulevel_3cat tertile m1_religion m1_occupation healthstatuss1_2cat m1_malnutrition m1_anemia_px m1_ipv_px_cat m1_trimester preg_intent anc_hrqol_2cat m1_dangersigns obstproblem_cat, by(dropout) // m1_marital excluded beacuse all are married, 
restore
   *Interpretation standardized difference(stddiff): if the absolute value of stddiff <0.1 well balanced, 0.1-0.2 small difference and >0.2 meaningful imbalance b/n completer and dropout

 *The standardized difference for the baseline characteristics was below 0.2, showing no difference between completers and dropouts. However, small differences were observed in women's wealth index (std diff=0.19) and a high difference in pregnancy intention (0.29), Woen HRQoL (0.31), and experienced danger signs(0.37) (SX table).  with a standardized difference of 0.19. All values were below the threshold of meaningful imbalance 0.2, indicating an acceptable baseline comparable to the SX table. The logistic regression analysis showed that both the baseline outcome variable and key covariates did not predict the dropout, which indicated a low risk of attrition bias (Sx Table).  
   
*Logistic regression
recode completer 0=1 1=0, gen(dropout)
lab def dropout 1"Yes" 0"No"
lab val dropout dropout
ta dropout

preserve
keep if time==1
logistic dropout ib0.depre_cat ib3.m1_age_cat1 ib2.m1_residence ib3.m1_edulevel_3cat ib3.tertile ib3.m1_religion ib4.m1_occupation ib0.healthstatuss1_2cat ib0.m1_malnutrition ib0.m1_anemia_px  ib1.preg_intent ib0.m1_ipv_px_cat ib1.anc_hrqol_2cat ib0.m1_dangersigns ib0.obstproblem_cat // ib0.m1_marital and ib3.m1_trimester not included because no value in 2by2 table
restore
*Interpretation logistic regression: if deprecate predict the outcome(completer), there is attrition bias. However, it is not predicting dropout (P=0.0.203) support the MAR assumption
*The result of logistic regression analysis showed that both the baseline outcome variable and other key covariates did not predict the dropout, which indicated a low risk of attrition bias. 

*Using GEE, considering the original data (full sample): Using seven time point data
xtset idn time
xtgee depre_cat ib3.m1_edulevel_3cat ib0.m1_marital ib3.tertile ib3.m1_religion /// 
ib3.trimesters ib1.preg_intent ib0.m1_ipv_px_cat ib1.healthstatus ///
ib1.anc_hrqol_2cat ib0.pxdanger_cat ib0. obstproblem_cat, ///
family(binomial 1) link(logit) corr(exc) vce(robust) eform
estimates store fullsample

*Now restrict to completers only:
xtgee depre_cat ib3.m1_edulevel_3cat ib3.tertile ib3.m1_religion /// 
ib3.trimesters ib1.preg_intent ib0.m1_ipv_px_cat ib1.healthstatus ///
ib1.anc_hrqol_2cat ib0.pxdanger_cat ib0. obstproblem_cat if completer==1, ///
family(binomial 1) link(logit) corr(exc) vce(robust) eform //ib0.m1_marital excluded because all are married. Additionally, ib7.time is not considered
estimates store completers

esttab fullsample completers, eform cells("b(fmt(2)) ci(fmt(2) par) p(fmt(3))") label //Showed OR + CI +p-value best

esttab fullsample completers, eform ///
cells("b(fmt(2)) ci(fmt(2) par)") ///
stats(N, fmt(0) labels("N")) label // Showed OR and CI
********************************************************************************

**-------------------------------------------------------------------------------
 "Completer vs drop out analsys:chi squre, logistic regression and sensitivity analysis"
                *Using 6-time opint dataset*
*-------------------------------------------------------------------------------
***Open the six-point long dataset
preserve
keep if !missing(depre_cat, time)
ta depre_cat
ta time
restore

*Count the number of observations per participant
keep if !missing(depre_cat, time)

**Estimate Dropout Percentage
*First create completer variable
bysort idn: gen n_rounds= _N 
gen completer = (n_rounds ==6)
label def completer 0 "Dropped out" 1 "Completed all 6"
label val completer completer
ta completer

recode completer 0=1 1=0, gen(dropout)
lab def dropout 1 "Yes" 0 "No"
lab values dropout dropout
ta dropout

*Now estimate dropout rate (baseline sample):
preserve
keep if time==1
tab completer 
tab dropout
restore

*Baseline comparison table
*Checking the presence significant difference between completers and dropouts using baseline characteristic variables and other covariates
*For binary baseline outcome:
preserve
keep if time==1 
tab depre_cat completer, row chi2 // p=0.000/significant d/t
ta m1_age_cat1 completer, row chi2 //p=0.061/No significant d/t
ta m1_marital completer, row chi2 //p=0.033/ only one obervation in 2by2 table
ta m1_residence completer, row chi2 //p=0.076
ta m1_edulevel_3cat completer, row chi2 //p=0.022
ta tertile completer, row chi2 // 0.000
ta m1_religion completer, row chi2 // p=0.882
ta m1_occupation completer, row chi2 // p=0.185
ta m1_residence completer, row ch //0.076
ta m1_health_literacy completer, row chi2 //p=0.996
ta healthstatuss1_2cat completer, row chi2 //0.039
ta m1_malnutrition completer, row chi2 // p=0.993
ta m1_anemia_px completer, row chi2 // p=0.412
ta m1_ipv_px_cat completer, row chi2 // p=0.532
ta m1_risk_health_px_cat completer, row chi2 // p=0.664
ta subuse_cat completer, row chi2 // p=0.562
ta m1_trimester completer, row chi2 // 0.000 // zero value in 2by2 table
ta preg_intent completer, row chi2 //0.000
ta anc_hrqol_2cat completer, row chi2 //p=0.018
ta anc1px_dangersign completer, row chi2 // p=0.325
ta obstproblem_cat completer, row chi2 // p=0.102
restore

*Optional: Standardized Differences (Stronger)
preserve
keep if time==1
//ssc install stddiff
stddiff depre_cat m1_age_cat1 m1_marital m1_residence m1_edulevel_3cat tertile m1_religion m1_occupation healthstatuss1_2cat m1_malnutrition m1_anemia_px m1_ipv_px_cat m1_trimester preg_intent anc_hrqol_2cat m1_dangersigns obstproblem_cat, by(dropout) // m1_marital, m1_trimester had 1 & 0 value in 2by2 table, respectively
restore

  *Interpretation standardized difference(stddiff): if the absolute value of stddiff <0.1 well balanced, 0.1-0.2 small difference and >0.2 meaningful imbalance b/n completer and dropout
   
 *The standardized difference for the baseline characteristics was below 0.2, showing no difference between completers and dropouts. However, small differences were observed in women's wealth index (std diff=0.19) and a high difference in pregnancy intention (0.29), Woen HRQoL (0.31), and experienced danger signs(0.37) (SX table).  with a standardized difference of 0.19. All values were below the threshold of meaningful imbalance 0.2, indicating an acceptable baseline comparable to the SX table. The logistic regression analysis showed that both the baseline outcome variable and key covariates did not predict the dropout, which indicated a low risk of attrition bias (Sx Table).  
   
*Logistic regression
recode completer 0=1 1=0, gen(dropout)
lab def dropout 1"Yes" 0"No"
lab val dropout dropout
ta dropout

preserve
keep if time==1
ta m1_trimester completer, ch
restore

preserve
keep if time==1
logistic dropout ib0.depre_cat ib3.m1_age_cat1 ib0.m1_marital ib2.m1_residence ib3.m1_edulevel_3cat ib3.tertile ib3.m1_religion ib4.m1_occupation ib0.healthstatuss1_2cat ib0.m1_malnutrition ib0.m1_anemia_px ib1.preg_intent ib0.m1_ipv_px_cat ib1.anc_hrqol_2cat ib0.m1_dangersigns ib0.obstproblem_cat 
restore //ib3.m1_trimester excuded beacuse it have 0 value in 2by2 table

*Interpretation logistic regression: if deprecate predict the outcome(completer), there is attrition bias. However, it is not predicting dropout (P=0.0.203) support the MAR assumption
*The result of logistic regression analysis showed that both the baseline outcome variable and other key covariates did not predict the dropout, which indicated a low risk of attrition bias. 

*Using GEE, considering the original data (full sample): Using seven time point data
xtset idn time
xtgee depre_cat ib3.m1_edulevel_3cat ib0.m1_marital ib3.tertile ib3.m1_religion /// 
ib3.trimesters ib1.preg_intent ib0.m1_ipv_px_cat ib1.healthstatus ///
ib1.anc_hrqol_2cat ib0.pxdanger_cat ib0. obstproblem_cat, ///
family(binomial 1) link(logit) corr(exc) vce(robust) eform
estimates store fullsample

*Now restrict to completers only:
xtgee depre_cat ib3.m1_edulevel_3cat ib0.m1_marital ib3.tertile ib3.m1_religion /// 
ib3.trimesters ib1.preg_intent ib0.m1_ipv_px_cat ib1.healthstatus ///
ib1.anc_hrqol_2cat ib0.pxdanger_cat ib0. obstproblem_cat if completer==1, ///
family(binomial 1) link(logit) corr(exc) vce(robust) eform //ib0.m1_marital excluded because all are married. Additionally, ib7.time is not considered
estimates store completers

esttab fullsample completers, eform cells("b(fmt(2)) ci(fmt(2) par) p(fmt(3))") label //Showed OR + CI +p-value best

esttab fullsample completers, eform ///
cells("b(fmt(2)) ci(fmt(2) par)") ///
stats(N, fmt(0) labels("N")) label // Showed OR and CI


