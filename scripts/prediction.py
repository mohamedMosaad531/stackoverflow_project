import pickle 
import yaml
import pandas as pd
import mlflow
from mlflow import MlflowClient
import os


LOG_DATA_PKL='data.pkl'
LOG_MODEL_PKL='model.pkl'
LOG_METRICS_PKL='metrics.pkl'



class JobPrediction:
    def __init__(self,track_url,run_id,feature_skills_clusters):
        self.track_url=track_url
        self.run_id=run_id

        self.feature_skills_clusters=feature_skills_clusters
        self.cluster_df=self.clusters()

        model_objs=self.load_mlflow_objs() 
        self.model=  model_objs[0]
        self.feature_names= model_objs[1]
        self.target_names=model_objs[2]


    def load_mlflow_objs(self):
        mlflow.set_tracking_uri(self.track_url)    
        client=MlflowClient()
        run=client.get_run(self.run_id)
        artifacts=run.info.artifact_uri
        artifacts=artifacts.replace("file:///c:/Users/Mohamed Mosaad/Downloads/first_git/data_science_project/notebooks/",'') 
        # print(artifacts)
        with open (os.path.join(artifacts,LOG_DATA_PKL),'rb')as aa:
            data_pkl=pickle.load(aa)
        # print (data_pkl)    
        with open (os.path.join(artifacts,LOG_MODEL_PKL),'rb') as bb:
            model_pkl=pickle.load(bb)

        # print(model_pkl)
        return model_pkl['model_object'],\
        data_pkl['features_names'],\
        data_pkl['targets_names']
    

    def clusters(self):
        with open (self.feature_skills_clusters,'r') as cc:
            clusters_with_skills=yaml.safe_load(cc)
        aa=[]
        bb=[]
        for cluster,skills in clusters_with_skills.items():
            # print(cluster,skills)
            for skill in skills:
                aa.append((skill,cluster))
        cluster_df=pd.DataFrame(aa,columns=["skills",'clusters'])
         # print(clusters_with_skills)    
        return cluster_df 
    
    def clusters_with_skills(self,input:list):
        all_features=self.feature_names
        # print(all_features)
        # print(len(all_features))

        new_cluster_df=self.cluster_df
        new_cluster_df["freq"]=new_cluster_df['skills'].isin(input)
        # print(new_cluster_df)
        # print(new_cluster_df.groupby('clusters')['freq'].sum())
        clusters=new_cluster_df.groupby('clusters')['freq'].sum()
        cluster_names=clusters.index


        all_features=pd.Series(all_features)
        skills_only=all_features[~all_features.isin(cluster_names)]
        # print(skills_only)
        ohe_skills=pd.Series(skills_only.isin(input).astype(int).to_list(),index=skills_only)
        # print(ohe_skills)

        prepared_features=pd.concat([ohe_skills,clusters])
        # print(prepared_features)
        # print(prepared_features.shape)

        return prepared_features
        # print(skills_only)
        
    def predict_jop(self,input):
        classifier=self.model
        prepared_features=[self.clusters_with_skills(input).values]
        # print(prepared_features)
        role_prediction=classifier.predict_proba(prepared_features)
        prediction=[prop[0][1] for prop in role_prediction ]
        prediction=pd.Series(prediction,index=self.target_names).sort_values(ascending=False)
        # print(prediction)
        return prediction
    
    def suggest_skills_for_role(self,skills,role,threshold=0):
        all_features=self.feature_names
        roles_for_skills=self.predict_jop(skills)
        cluster_names=self.cluster_df['clusters'].unique()
        all_features=pd.Series(all_features)
        skills_only=all_features[~all_features.isin(cluster_names)]
        # print(roles_for_skills["Data or business analyst"])
        skills_needed=[]
        for skill in skills_only:
            new_skills=skills.copy()
            new_skills.append(skill)
            diff=(self.predict_jop(new_skills)[role]-roles_for_skills[role])/roles_for_skills[role]
            if diff>threshold:
                skills_needed.append(skill)

        return skills_needed        
            

        