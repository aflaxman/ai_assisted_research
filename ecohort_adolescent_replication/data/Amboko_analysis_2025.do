use "C:\Users\bamboko\OneDrive - Kemri Wellcome Trust\HERU\Kenya\MNH e-cohort\Manuscripts\Adolescent paper analysis\latest\Data\v1eco_m1-m3_et_wide_der_subset.dta", clear

*Labe define
label define m3_619hlabel 0 "No" 1 "Yes"
label val m3_619h m3_619hlabel

*countrylabel
rename country country2
gen country=1 if country2=="Ethiopia"
replace country=2 if country2=="Kenya"
replace country=3 if country2=="South Africa"
ta country

label define countrylabel 1 "Ethiopia" 2 "Kenya" 3 "South Africa"
label val country countrylabel

clonevar site=study_site
svyset site || facility, 
svyset(site)

*1=adolescent
gen age_cat_num= enrollage<=19
replace age_cat_num=. if enrollage==.
replace age_cat_num=2 if age_cat_num==0

ta age_cat_num
gen age_group=age_cat_num==1
replace age_group=. if age_cat_num==.
ta age_group

*South Africa
clonevar m1_enrollage=enrollage
gen adole_age=m1_enrollage>=15 & m1_enrollage<=17
replace adole_age=2 if m1_enrollage>17 & m1_enrollage<=19
*replace adole_age=2 if m1_enrollage>19
replace adole_age=. if m1_enrollage==.
ta adole_age age_cat_num, col

**DEMOGRAPHICS
*age
summ m1_enrollage if age_cat_num==1
summ m1_enrollage if age_cat_num==2
summ m1_enrollage 

*marital status
ta marriedp age_cat_num, col

svy:ta marriedp age_cat_num, col per

*education
ta educ_cat age_cat_num, col 

gen educ_level=educ_cat==3
replace  educ_level=. if educ_cat==.
ta educ_level age_cat_num, col
svy: ta educ_level age_cat_num, col per

*KE & SA
gen educ_level=educ_cat==3 | educ_cat==4
replace  educ_level=. if educ_cat==.
ta educ_level age_cat_num, col

**all countries
gen educ_level2= 0 if educ_cat==1
replace  educ_level2=1 if educ_cat==2
replace  educ_level2=2 if educ_cat==3 | educ_cat==4
replace  educ_level2=. if educ_cat==.
ta educ_level2 age_cat_num, col


*wealth
ta quintile
gen wealth_cat= quintile==2 | quintile==3 | quintile==4
replace wealth_cat=2 if quintile==5
replace wealth_cat=. if quintile==.
ta  wealth_cat

*South Africa
ta tertile age_cat_num, col

gen wealth_cat= tertile==2 | tertile==3 
*replace wealth_cat=2 if quintile==5
replace wealth_cat=. if tertile==.
ta  wealth_cat age_cat_num, col

replace wealth_cat=0 if tertile==1 & country==3
replace wealth_cat=1 if tertile==2 & country==3
replace wealth_cat=2 if tertile==3 & country==3
ta  wealth_cat age_cat_num, col

*Region, 0=urban, 1=rural
*South Africa
ta region age_cat_num,col

gen location = region==2
ta location age_cat_num,col

*Kenya
gen location = region=="Rural"
ta location age_cat_num,col

*literacy score
gen literacy_level= m1_health_literacy==4
ta literacy_level age_cat_num, col

*SA
gen literacy_level=health_literacy==4
ta literacy_level age_cat_num, col

*employment
gen employ_status=employment==1
replace employ_status=. if missing(employment)
ta employ_status age_cat_num, col

*South Africa
gen employment=1 if m1_506==6
replace employment=2 if m1_506==7
replace employment=3 if m1_506==10
replace employment=4 if m1_506==
replace employment=0 if employment==. & m1_506!=.

ta employment age_cat_num, col

gen employ_status=employment==0
replace employ_status=. if missing(employment)
ta employ_status age_cat_num, col

*social support
clonevar social_support=m1_508
summ social_support if age_cat_num==1
summ social_support if age_cat_num==2
summ social_support

*Gravidity
summ gravidity if age_cat_num==1
summ gravidity if age_cat_num==2 
summ gravidity

gen primi_gravida=gravidity==1
ta primi_gravida age_cat_num, col

*gestation at first ANC
ta m1_trimester age_cat_num, col

gen firstANCvisit_1trimester=m1_trimester==1
replace firstANCvisit_1trimester=. if m1_trimester==.
ta firstANCvisit_1trimester age_cat_num, col m
svy:tab firstANCvisit_1trimester age_group, col per

*South Africa
clonevar m1_trimester = trimester
ta m1_trimester age_cat_num, col

gen firstANCvisit_1trimester=m1_trimester==1
replace firstANCvisit_1trimester=. if m1_trimester==.

ta firstANCvisit_1trimester age_cat_num, col m
svy:tab firstANCvisit_1trimester age_group, col per

*pregnancy intentionality
ta preg_intent age_cat_num, col

**medical history
*MUAC/BMI
*SA
ta low_BMI age_cat_num, col

clonevar lowBMI_MUAC=low_BMI
ta lowBMI_MUAC age_cat_num, col

*ET
br anc1bmi m1_BMI m1_low_BMI m1_muac anc1muac m1_malnutrition

gen lowBMI_MUAC=m1_low_BMI==1 | m1_malnutrition==1
ta lowBMI_MUAC age_cat_num, col

*KE
clonevar lowBMI_MUAC=bsl_low_BMI
ta lowBMI_MUAC age_cat_num, col

*danger sign
ta m1_dangersigns age_cat_num,col

*SA
clonevar m1_dangersigns=dangersigns
ta m1_dangersigns age_cat_num,col

*depression
ta ANCdepression age_cat_num, col
*SA
clonevar ANCdepression=anc1depression
ta ANCdepression age_cat_num, col

*overall health rating
ta m1_overall_health age_cat_num, col

**module 1 overall health rating 
*poor/fair SA/ET
gen m1_overall_health=m1_201==4 | m1_201==5
tab m1_overall_health age_cat_num,col

*risky behavior
ta m1_risk_health age_cat_num, col

*SA 
clonevar m1_risk_health=risk_health
ta m1_risk_health age_cat_num, col

*abuse
gen violence_abuse=m1_1101==1 | m1_1103==1
ta violence_abuse age_cat_num, col

**CONTENT OF CARE
**Vital signs

*blood pressure
gen bp_taken=m1_700==1
replace bp_taken=. if m1_700==.
ta bp_taken age_cat_num,col

*weighed
gen weight_taken=m1_701==1
replace weight_taken=. if m1_701==.
ta weight_taken age_cat_num,col

*height
gen height_taken=m1_702==1
replace height_taken=. if m1_702==.
ta height_taken age_cat_num,col

*muac
gen muac_taken= m1_703==1
replace muac_taken=. if  m1_703==.
ta muac_taken age_cat_num,col

*investigations
*HIV
gen hiv_test=m1_708a==1
replace hiv_test=. if m1_708a==.
ta hiv_test age_cat_num,col

*syphillis
gen syphillis_test=m1_710a==1
replace syphillis_test=. if m1_710a==.
ta syphillis_test age_cat_num,col

*blood sugar
gen bs_test=m1_711a==1
replace bs_test=. if m1_711a==.
ta bs_test age_cat_num,col

*urinalysis
gen urine_test= m1_705==1
replace urine_test=. if missing(m1_705)
ta urine_test age_cat_num,col

*Hb
gen hb_taken=m1_1307>0
replace hb_taken=0 if missing(m1_1307)
ta hb_taken age_cat_num, col

*ultrasound
gen ultras_done=m1_712==1
replace ultras_done=0 if missing(m1_712)
ta ultras_done age_cat_num, col

**supplements & treatments
*IFA
gen m1_given_IFA= m1_713a==1
replace m1_given_IFA=. if missing(m1_713a)
ta m1_given_IFA age_cat_num, col

*Calcium
gen m1_given_calcium= m1_713b==1
replace m1_given_calcium=. if missing(m1_713b)
ta m1_given_calcium age_cat_num, col

*tetanus
gen m1_given_tt= m1_714a==1
replace m1_given_tt=. if missing(m1_714a)
ta m1_given_tt age_cat_num, col

*KE
gen m1_given_tt= anc1tt==1
replace m1_given_tt=. if missing(anc1tt)
ta m1_given_tt age_cat_num, col

**First ANC completeness score
*assessment and tests
egen FirstANC_completeness= rowmean(m1_700 m1_701 m1_702 m1_703  m1_708a m1_710a m1_711a urine_test hb_taken m1_712 m1_given_IFA m1_given_calcium m1_given_tt)

summ FirstANC_completeness if age_cat_num==1
summ FirstANC_completeness if age_cat_num==2

svy: mean FirstANC_completeness, over(age_cat_num)
svy: mean FirstANC_completeness, over(age_cat_num) coeflegend

test  _b[c.FirstANC_completeness@1bn.age_cat_num] =_b[c.FirstANC_completeness@2.age_cat_num]

*KE
egen FirstANC_completeness_ke= rowmean(m1_700 m1_701 m1_702 m1_703  m1_708a m1_710a m1_711a urine_test hb_taken m1_712 m1_given_IFA m1_given_tt)

rename FirstANC_completeness_ke FirstANC_completeness, replace 

summ FirstANC_completeness if age_cat_num==1
summ FirstANC_completeness if age_cat_num==2

svy: mean FirstANC_completeness, over(age_cat_num)
svy: mean FirstANC_completeness, over(age_cat_num) coeflegend

test  _b[c.FirstANC_completeness@1bn.age_cat_num] =_b[c.FirstANC_completeness@2.age_cat_num]

*counselling
egen firstANC_counsel_comp=rowmean(m1_716e m1_716a m1_716b m1_809 m1_724a)

summ firstANC_counsel_comp if age_cat_num==1
summ firstANC_counsel_comp if age_cat_num==2

svy: mean firstANC_counsel_comp, over(age_cat_num)
svy: mean firstANC_counsel_comp, over(age_cat_num) coeflegend

test   _b[c.firstANC_counsel_comp@1bn.age_cat_num] =_b[c.firstANC_counsel_comp@2.age_cat_num]

*KE
egen firstANC_counsel_comp_ke=rowmean(m1_716e m1_716a m1_716b anc1counsel_birthplan m1_724a)

rename firstANC_counsel_comp_ke firstANC_counsel_comp, replace

summ firstANC_counsel_comp_ke if age_cat_num==1
summ firstANC_counsel_comp_ke if age_cat_num==2

svy: mean firstANC_counsel_comp_ke, over(age_cat_num)
svy: mean firstANC_counsel_comp_ke, over(age_cat_num) coeflegend

test   _b[c.firstANC_counsel_comp_ke@1bn.age_cat_num] =_b[c.firstANC_counsel_comp_ke@2.age_cat_num]

*1st ANC visit quality
ta vgm1_601
gen m1_ANC1visit_qual= vgm1_601==1
ta m1_ANC1visit_qual age_cat_num, col

svy:ta m1_ANC1visit_qual age_cat_num, col per

*FU ANC visits
gen FU_ANCvisits_total=n_anc_r1 + n_anc_r2 + n_anc_r3 + n_anc_r4 + n_anc_r5 + n_anc_r6 + n_anc_r7

summ FU_ANCvisits_total if age_cat_num==1
summ FU_ANCvisits_total if age_cat_num==2

svy: mean FU_ANCvisits_total, over(age_cat_num)
svy: mean FU_ANCvisits_total, over(age_cat_num) coeflegend

test   _b[c.FU_ANCvisits_total@1bn.age_cat_num] =_b[c.FU_ANCvisits_total@2.age_cat_num]

*SA
egen m2_ancvisits_r1=rowtotal(m2_305_r1  m2_308_r1  m2_311_r1  m2_314_r1  m2_317_r1)

egen m2_ancvisits_r2=rowtotal(m2_305_r2  m2_308_r2  m2_311_r2  m2_314_r2  m2_317_r2)

egen m2_ancvisits_r3=rowtotal(m2_305_r3  m2_308_r3  m2_311_r3  m2_314_r3  m2_317_r3)

egen m2_ancvisits_r4=rowtotal(m2_305_r4  m2_308_r4  m2_311_r4  m2_314_r4  m2_317_r4)

egen m2_ancvisits_r5=rowtotal(m2_305_r5 m2_308_r5 m2_311_r5  m2_314_r5  m2_317_r5)

egen m2_ancvisits_r6=rowtotal(m2_305_r6 m2_308_r6 m2_311_r6  m2_314_r6  m2_317_r6)

egen FU_ANCvisits_total=rowtotal(m2_ancvisits_r1 m2_ancvisits_r2 m2_ancvisits_r3 m2_ancvisits_r4 m2_ancvisits_r5 m2_ancvisits_r6)

summ FU_ANCvisits_total if age_cat_num==1
summ FU_ANCvisits_total if age_cat_num==2

svy: mean FU_ANCvisits_total, over(age_cat_num)
svy: mean FU_ANCvisits_total, over(age_cat_num) coeflegend

test   _b[c.FU_ANCvisits_total@1bn.age_cat_num] =_b[c.FU_ANCvisits_total@2.age_cat_num]

*total ANC visits

*
summ anc_total if age_cat_num==1
summ anc_total if age_cat_num==2

gen anc4_visits=anc_total>=4


*SA
gen anc_total= FU_ANCvisits_total +1

summ anc_total if age_cat_num==1
summ anc_total if age_cat_num==2

svy: mean anc_total, over(age_cat_num)
svy: mean anc_total, over(age_cat_num) coeflegend

test   _b[c.anc_total@1bn.age_cat_num] =_b[c.anc_total@2.age_cat_num]

gen anc4_visits=anc_total>=4

ta anc4_visits age_cat_num, col
ta anc4_visits age_cat_num, col chi2

*atleast one FU blood test
gen atleast1_FU_bloodtest=m2_501d_r1==1 | m2_501d_r2==1 | m2_501d_r3==1 | m2_501d_r4==1 | m2_501d_r5==1 | m2_501d_r6==1 
ta  atleast1_FU_bloodtest age_cat_num, col

ta atleast1_FU_bloodtest age_cat_num, col chi2

*atleast one FU urine test
gen atleast1_FU_urinetest=m2_501e_r1==1 |  m2_501e_r2==1 |  m2_501e_r3 ==1 | m2_501e_r4==1 |  m2_501e_r5==1 |  m2_501e_r6==1 
ta  atleast1_FU_urinetest age_cat_num, col

ta atleast1_FU_urinetest age_cat_num, col chi2

*food supplementation
gen atleast1_foodsup=anc1food_supp==1 | m2_601d_r1==1 | m2_601d_r2==1 | m2_601d_r3==1 | m2_601d_r4==1 | m2_601d_r5==1 | m2_601d_r6==1 
ta  atleast1_foodsup age_cat_num, col chi2

ta atleast1_foodsup age_cat_num, col per

*KE
gen atleast1_foodsup=anc1food_supp==1 | m2_601d_r1==1 | m2_601d_r2==1 | m2_601d_r3==1 | m2_601d_r4==1 | m2_601d_r5==1 | m2_601d_r6==1 | m2_601d_r7==1 | m2_601d_r8==1 | m2_601d_r9==1 
ta  atleast1_foodsup age_cat_num, col chi2

ta atleast1_foodsup age_cat_num, col per

*atleast one ultrasound done
*SA
gen atleast1_ultrasound= m1_712==1 | m2_501f_r1==1 | m2_501f_r2==1 | m2_501f_r3==1 | m2_501f_r4==1 | m2_501f_r5==1 | m2_501f_r6==1 
ta  atleast1_ultrasound age_cat_num, col

svy:ta atleast1_ultrasound age_cat_num, col per

*Adherence to folic
gen FU_folic=m2_603_r1==1 | m2_603_r2==1 | m2_603_r3==1 |  m2_603_r4==1 | m2_603_r5==1 | m2_603_r6==1 |  m2_603_r7==1 |  m2_603_r8==1
ta FU_folic age_cat_num, col

svy:ta FU_folic age_cat_num, col per

*SA
gen FU_folic=m2_603_r1==1 | m2_603_r2==1 | m2_603_r3==1 |  m2_603_r4==1 | m2_603_r5==1 | m2_603_r6==1 

ta FU_folic age_cat_num, col chi2

svy:ta FU_folic age_cat_num, col per

*FP counselling
*ET
gen atleast1_FP_counsel=m2_506d_r1==1 | m2_506d_r3==1 | m2_506d_r5==1 | m2_506d_r6==1 | m2_506d_r7==1 |m2_506d_r8==1 

ta atleast1_FP_counsel age_cat_num, col chi2

*KE
gen atleast1_FP_counsel=m2_506d_r1==1 | m2_506d_r2==1 | m2_506d_r3==1 | m2_506d_r4==1 | m2_506d_r5==1 | m2_506d_r6==1 | m2_506d_r7==1 | m2_506d_r8==1 | m2_506d_r9==1 

ta atleast1_FP_counsel age_cat_num, col chi2
*SA
gen atleast1_FP_counsel=m2_506d_r1==1 | m2_506d_r2==1 | m2_506d_r3==1 | m2_506d_r4==1 | m2_506d_r5==1 | m2_506d_r6==1 

ta atleast1_FP_counsel age_cat_num, col chi2
 

***delivery care
*privacy
gen delivery_privacy=m3_604b==1
replace delivery_privacy=. if missing(m3_604b)
ta delivery_privacy age_cat_num, col

svy:ta delivery_privacy age_cat_num, col per

*mistreatment 
gen del_mistreatment=m3_1005a==1 | m3_1005b==1 |  m3_1005c==1 |  m3_1005d==1 |  m3_1005e==1 |  m3_1005f==1 |  m3_1005g==1 |  m3_1005h==1 
replace del_mistreatment=. if missing(m3_1005a) | missing(m3_1005b) |missing(m3_1005c) | missing(m3_1005d) | missing(m3_1005e) | missing(m3_1005f) | missing(m3_1005g) | missing(m3_1005h)
ta del_mistreatment age_cat_num, col 

ta del_mistreatment age_cat_num, col chi2

**live births-used this to restrict module 3 data
ta m3_303b
gen m3_baby_alive=m3_303b==1
replace m3_baby_alive=. if m3_303b!=1
ta m3_baby_alive

**quality of care rating
*overall rating delivery qoc
ta m3_1001_QOC 
gen del_overall_qual=m3_1001_QOC==1
replace del_overall_qual=0 if m3_1001_QOC==4
replace del_overall_qual=. if m3_1001_QOC==. | (m3_1001_QOC!=1 & m3_1001_QOC!=2 & m3_1001_QOC!=3 & m3_1001_QOC!=4)
ta del_overall_qual age_cat_num if m3_baby_alive==1, col

*SA
ta m3_1001 
gen del_overall_qual=m3_1001==4 | m3_1001==5 
*replace del_overall_qual=0 if m3_1001_QOC==4
*replace del_overall_qual=. if m3_1001_QOC==. | (m3_1001_QOC!=1 & m3_1001_QOC!=2 & m3_1001_QOC!=3 & m3_1001_QOC!=4)
ta del_overall_qual age_cat_num if m3_baby_alive==1, col

svy:ta del_overall_qual age_cat_num if m3_baby_alive==1, col per

*KE
ta m3_1001 
gen del_overall_qual_ke=m3_1001==4 | m3_1001==5 
*replace del_overall_qual=1 if m3_1001==4
replace del_overall_qual_ke=. if m3_1001==. | (m3_1001!=1 & m3_1001!=2 & m3_1001!=3 & m3_1001!=4 & m3_1001!=5)
ta del_overall_qual_ke age_cat_num if m3_baby_alive==1, col

rename del_overall_qual_ke del_overall_qual, replace

*specific aspects of care
ta m3_1004
 

*hosp delivery
m3_501, m3_502, m3_510

*delivery at a HF
ta m3_501
gen m3_del_HF= m3_501==1
replace m3_del_HF=. if m3_501!=1 & m3_501!=0
ta m3_del_HF age_cat_num, col

ta m3_del_HF age_cat_num, col  chi2

*delivery facility ownership 
ta m3_502

gen del_HF_ownership= m3_502==3 | m3_502==4 | m3_502==5 | m3_502==6

replace del_HF_ownership=2 if m3_502==8 |m3_502==9 

replace del_HF_ownership=. if m3_502==. | (m3_502!=3 & m3_502!=4 & m3_502!=5 & m3_502!=6 & m3_502!=7 & m3_502!=8 & m3_502!=9)
ta del_HF_ownership age_cat_num,col

*SA
gen del_HF_ownership= m3_502==3 | m3_502==4 | m3_502==7 
replace del_HF_ownership=2 if m3_502==6 

replace del_HF_ownership=. if m3_502==. | (m3_502!=3 & m3_502!=4 & m3_502!=6 & m3_502!=7)
ta del_HF_ownership age_cat_num,col

*delivery HF level
gen del_HF_level= m3_502==2 | m3_502==5

replace del_HF_level=. if m3_502==. | (m3_502!=3 & m3_502!=4 & m3_502!=5 & m3_502!=6 & m3_502!=7 & m3_502!=8 & m3_502!=9)
ta del_HF_level age_cat_num, col

*SA
gen del_HF_level= m3_502==3 | m3_502==7 
replace del_HF_level=. if m3_502==. | (m3_502!=3 & m3_502!=4 & m3_502!=6 & m3_502!=7 )
ta del_HF_level age_cat_num, col

*visited another HF before
ta m3_510
gen m3_another_HF=m3_510==1
replace m3_another_HF=. if m3_510!=1 & m3_510!=0
ta m3_another_HF age_cat_num,col

*vaginal examination
ta m3_1006a
gen m3_vag_exam=m3_1006a==1
ta m3_vag_exam age_cat_num if m3_del_HF==1, col chi2

svy: ta m3_vag_exam age_cat_num if m3_del_HF==1, col per

*permission
ta m3_1006b
gen m3_vag_exam_per=m3_1006b==1
ta m3_vag_exam_per age_cat_num if m3_del_HF==1, col

svy: ta m3_vag_exam_per age_cat_num if m3_del_HF==1, col per

*privacy
ta m3_1006c
gen m3_vag_exam_privacy=m3_1006c==1
ta m3_vag_exam_privacy age_cat_num if m3_del_HF==1, col

svy: ta m3_vag_exam_privacy age_cat_num if m3_del_HF==1, col per

*pain relief
ta m3_1007b
gen m3_req_pain_relief=m3_1007b==1
ta m3_req_pain_relief age_cat_num if m3_del_HF==1, col

svy: ta m3_req_pain_relief age_cat_num if m3_del_HF==1, col per

*
ta m3_1007a
gen m3_pain_relief=m3_1007a==1 | m3_1007c==1
ta m3_pain_relief age_cat_num if m3_del_HF==1, col

svy: ta m3_pain_relief age_cat_num if m3_del_HF==1, col per


***pregnancy outcomes***
*cs
ta m3_605a age_cat_num, col
gen c_section=m3_605a==1
replace c_section=. if m3_605a==. | (m3_605a!=0 & m3_605a!=1)
ta c_section age_cat_num if m3_del_HF==1, col chi

ta c_section age_cat_num if m3_del_HF==1, col chi2

*episiotomy
ta m3_606
gen episiotomy_done=m3_606==1
*replace episiotomy_done=. if m3_606!=1 & m3_606!=0
ta  episiotomy_done age_cat_num if m3_del_HF==1, col chi

ta  episiotomy_done age_cat_num if m3_del_HF==1, col chi2

*prolonged labor
ta m3_704e
gen prolonged_labor=m3_704e==1
replace prolonged_labor=. if m3_704e!=1 & m3_704e!=0
ta prolonged_labor age_cat_num if m3_del_HF==1, col

ta prolonged_labor age_cat_num if m3_del_HF==1, col chi2

*ICU
ta m3_706
gen icu_adm=m3_706==1 
replace icu_adm=. if m3_706!=1 & m3_706!=0
ta icu_adm age_cat_num, col

ta icu_adm age_cat_num if m3_del_HF==1, col chi2

*Blood tranfussion
ta m3_705
gen blood_trans=m3_705==1
replace blood_trans=. if m3_705!=1 & m3_705!=0

ta blood_trans age_cat_num if m3_del_HF==1, col chi2

*Fistula
ta m3_805
gen m3_fistula=m3_805==1
replace m3_fistula=. if m3_805!=1 & m3_805!=0

ta  m3_fistula age_cat_num if m3_del_HF==1, col chi2

*severe health problems 
ta m3_702 
ta m3_703
clonevar m3_severe_health = m3_703
ed m3_702 m3_703 m3_severe_health
replace m3_severe_health=. if m3_705!=1 & m3_705!=0

ta m3_severe_health age_cat_num, col chi2 m

gen atleast1_obs_outcome= prolonged_labor==1 | icu_adm==1 | blood_trans==1 | m3_fistula==1 | m3_severe_health==1
ta atleast1_obs_outcome age_cat_num if m3_del_HF==1, col chi2

*Depression
gen pq2_score= (m3_801a + m3_801b)

gen depression_pp=pq2_score>=3
replace depression_pp=. if pq2_score==.
ta depression_pp age_cat_num if m3_del_HF==1, col chi2

*postnatal check-up before discharge
ta m3_613
gen m3_postnatal_checkup= m3_613==1
replace m3_postnatal_checkup=. if m3_613!=0 & m3_613!=1
ta m3_postnatal_checkup age_cat_num, col

ta m3_postnatal_checkup age_cat_num, col chi2

*continuity of care
ta firstANCvisit_1trimester age_cat_num, col chi2

ta anc4_visits age_cat_num, col chi2

ta m3_del_HF age_cat_num, col chi2 

ta m3_postnatal_checkup age_cat_num, col chi2

gen continuity_of_care=firstANCvisit_1trimester==1 & anc4_visits==1 & m3_del_HF==1 & m3_postnatal_checkup==1
ta continuity_of_care age_cat_num, col chi2

egen continuity_of_care_index= rowmean (firstANCvisit_1trimester anc4_visits m3_del_HF m3_postnatal_checkup)

summ continuity_of_care_index if age_cat_num==1
summ continuity_of_care_index if age_cat_num==2

svy: mean continuity_of_care_index, over(age_cat_num)
svy: mean continuity_of_care_index, over(age_cat_num) coeflegend

test   _b[c.continuity_of_care_index@1bn.age_cat_num] =_b[c.continuity_of_care_index@2.age_cat_num]

 *postpartum counselling

egen immediate_pp_counsel_comp=rowmean(m3_619a m3_619b m3_619c m3_619d m3_619e m3_619g m3_619h)

summ immediate_pp_counsel_comp if age_cat_num==1
summ immediate_pp_counsel_comp if age_cat_num==2

svy: mean immediate_pp_counsel_comp, over(age_cat_num)
svy: mean immediate_pp_counsel_comp, over(age_cat_num) coeflegend

test   _b[c.immediate_pp_counsel_comp@1bn.age_cat_num] =_b[c.immediate_pp_counsel_comp@2.age_cat_num]

*low birthweight
gen low_birth_wt= m3_baby1_weight<2.5 | m3_baby2_weight<2.5
ta low_birth_wt age_cat_num, col


*very likely to recommend provider
ta m3_1002
gen m3_recommend_provider=m3_1002==1
replace m3_recommend_provider= . if m3_1002==. | (m3_1002!=1 & m3_1002!=2 & m3_1002!=3 & m3_1002!=4)
ta m3_recommend_provider age_cat_num,col chi2

