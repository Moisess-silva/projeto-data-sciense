"""Execução unificada das Questões 1, 2 e 3.

Gera resultados, gráficos em assets/ e exporta o pipeline TPOT em outputs/.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, StackingClassifier

ROOT = Path(__file__).resolve().parent
ASSETS, OUTPUTS = ROOT / "assets", ROOT / "outputs"
SEED = 42
ASSETS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

def q1():
    d = load_wine(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=.2, stratify=d.target, random_state=SEED)
    sc = StandardScaler(); Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte); Xa = sc.transform(d.data)
    lda = LinearDiscriminantAnalysis(n_components=2); Ltr, Lte, La = lda.fit_transform(Xtr,ytr), lda.transform(Xte), lda.transform(Xa)
    pca = PCA(2, random_state=SEED); Ptr, Pte, Pa = pca.fit_transform(Xtr), pca.transform(Xte), pca.transform(Xa)
    def acc(a,b):
        m=KNeighborsClassifier(5).fit(a,ytr); return accuracy_score(yte,m.predict(b))
    la, pa = acc(Ltr,Lte), acc(Ptr,Pte)
    for vals, name, title in [(Pa,"q1_pca_scatter.png","PCA — projeção dos vinhos"),(La,"q1_lda_scatter.png","LDA — separação das classes")]:
        plt.figure(figsize=(9,6)); markers=["o","s","X"]; colors=["#e15759","#59a14f","#4c78a8"]; labels=["Classe 0","Classe 1","Classe 2"]
        for cls,marker,color,label in zip(range(3),markers,colors,labels):
            mask=d.target==cls; plt.scatter(vals[mask,0],vals[mask,1],marker=marker,c=color,label=label,s=75,edgecolors="white",linewidths=.5)
        plt.title(title,weight="bold"); plt.legend(title="Classe do vinho"); plt.tight_layout(); plt.savefig(ASSETS/name,dpi=300); plt.close()
    plt.figure(figsize=(7,5)); bars=plt.bar(["PCA","LDA"],[pa,la],color=["#4C78A8","#F58518"]); plt.ylim(0,1.08); plt.ylabel("Acurácia"); plt.title("KNN (k=5) — comparação",weight="bold"); [plt.text(b.get_x()+b.get_width()/2,b.get_height()+.02,f"{b.get_height():.1%}",ha="center",weight="bold") for b in bars]; plt.tight_layout(); plt.savefig(ASSETS/"q1_accuracy_comparison.png",dpi=300); plt.close()
    return {"PCA":pa,"LDA":la}

def q2():
    X,y=make_classification(n_samples=1200,n_features=6,n_informative=4,n_redundant=1,weights=[.62,.38],class_sep=1.15,random_state=SEED); rng=np.random.default_rng(SEED); df=pd.DataFrame(X,columns=[f"num_{i}" for i in range(1,7)]); df["faixa_renda"]=pd.qcut(df.num_1,3,labels=["baixa","media","alta"]); df["regiao"]=rng.choice(["Norte","Sul","Sudeste","Nordeste"],len(df)); nums=[f"num_{i}" for i in range(1,7)]; cats=["faixa_renda","regiao"]; prep=ColumnTransformer([("num",StandardScaler(),nums),("cat",OneHotEncoder(handle_unknown="ignore"),cats)])
    base=[("Regressão Logística",LogisticRegression(max_iter=2000)),("Árvore de Decisão",DecisionTreeClassifier(max_depth=5,random_state=SEED)),("SVM",SVC(kernel="rbf",probability=True,random_state=SEED))]; est=[(n,m) for n,m in base]; models=base+[("Voting",VotingClassifier(est,voting="hard")),("Stacking",StackingClassifier(est,final_estimator=LogisticRegression(max_iter=2000),cv=5))]; Xt,Xv,yt,yv=train_test_split(df,y,test_size=.2,stratify=y,random_state=SEED); rows=[]
    for n,m in models:
        p=Pipeline([("prep",prep),("model",m)]).fit(Xt,yt); pred=p.predict(Xv); rows.append({"Modelo":n,"Acurácia":accuracy_score(yv,pred),"Precision":precision_score(yv,pred),"Recall":recall_score(yv,pred),"F1-Score":f1_score(yv,pred)})
    r=pd.DataFrame(rows).set_index("Modelo"); r.plot(kind="bar",figsize=(12,6),color=["#4C78A8","#F58518","#54A24B","#E45756"]); plt.ylim(0,1.08); plt.title("Crédito — comparação de métricas",weight="bold"); plt.tight_layout(); plt.savefig(ASSETS/"q2_metrics_comparison.png",dpi=300); plt.close(); r.round(4).to_csv(OUTPUTS/"q2_results.csv"); return r

def q3():
    from tpot import TPOTClassifier
    d=load_breast_cancer(); Xt,Xv,yt,yv=train_test_split(d.data,d.target,test_size=.2,stratify=d.target,random_state=SEED); base=Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(max_iter=2000))]).fit(Xt,yt); ba=accuracy_score(yv,base.predict(Xv)); t=TPOTClassifier(generations=5,population_size=20,cv=5,random_state=SEED,verbosity=2,n_jobs=1).fit(Xt,yt); ta=accuracy_score(yv,t.predict(Xv)); t.export(str(OUTPUTS/"q3_tpot_best_pipeline.py")); return {"Baseline":ba,"TPOT":ta,"Pipeline":str(t.fitted_pipeline_)}

if __name__ == "__main__":
    print("Q1:", q1()); print("Q2:\n", q2()); print("Q3:", q3())

