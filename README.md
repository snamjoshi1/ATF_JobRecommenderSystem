# ATF_JobRecommenderSystem

Execution Steps:
1. Install all the libraries mentioned in the below files
2. Load required libraries as mentioned at the start of each file
3. Code under Classification is trained on GPU using google colab
4. Model Predictions are being carried out on CPU as GPU facility is unavailable


5. Start with folder Job Description:
	a. Data for Data Analysis and EDA is available under data\collected_data folder
	Data File: indeed_job_dataset.csv
	b. File Execution Sequence
		i) JD Preprocessing & EDA.ipynb: This will Produce a file df_description_processed.csv. This will be used as input for next set of files
		ii) POS Analysis And EDA_JD.ipynb
		iii) Regex_Chunking_JD.ipynb: This file will be executed on Google colab and model saved will be downloaded. this will serve as an input to next file
		iV) BuildingTrainingDataset_JD.ipynb : This will help getting phrase file and we name the file as train_skills.csv and this file can be found in \data\ folder under Job Description folder
	

6. Start with folder Profile: 
	a. Data for Data Analysis and EDA is available under data\collected_data folder
	Data File: disability_profiles.xlsx
	b. File Execution Sequence
		i) BioSkillsPreprocessing&EDA.ipynb: This will Produce a file df_processed_bio.csv. This will be used as input for next set of files
		ii) POS Analysis And EDA Profile.ipynb
		iii) Regex_Chunking_Profiles.ipynb: This file will be executed on Google colab and model saved will be downloaded. this will serve as an input to next file
		iV) BuildingTrainingDataset_Profiles.ipynb : This will help getting phrase file and we name the file as profile_skills.csv and this file can be found in \data\ folder under Job Description folder
	

7. We then manually label the data 
	i) skill vs non skill for JD and the final file is present under Classification\JD_Skills\data folder. File name is train_skills.csv
	ii) skill vs non skill vs education for profiles and the final file is present under Classification\Profile_Skills\data folder. File name is bio_train_skills_10k.xlsx

8. We then run the classification models for JD. File is FindSkillsFromJD.ipynb under Classification\JD_Skills and the final file we get as an output can be found in Output Dataset folder under classification.

9. We then run the classification models for Profile. File is GetSkillsFromDisabilityProfiles.ipynb under Classification\Profile_Skills and the final file we get as an output can be found in Output Dataset folder under classification.

10. Files obtained from step 8 and step 9 will be used as input for RecommendationEngine.ipynb. This file can be found under Recommendation folder

11. API code can be found under Deployment folder.

12. code to host API:--Use Anaconda Command Prompt
	1. Go to Respective environment where all required libraries are installed: conda activate <env name>
	For Skills from Description
	1. Go to the folder Deployment\JobDescription_Skill_API
	2. execute uvicorn SkillPrediction:app --reload
	For Jobs from Bio
	1. Go to the folder Deployment\RecommendationAPI
	2. execute uvicorn main:app --reload
