from pathlib import Path
import json, urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"; OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/dataset_part_2.csv"
CSV = DATA / "dataset_part_2.csv"

def load_data():
    if not CSV.exists():
        print("Downloading official IBM Skills Network dataset...")
        urllib.request.urlretrieve(URL, CSV)
    df = pd.read_csv(CSV)
    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' was not found.")
    return df

def main():
    df = load_data()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["PayloadMass"] = pd.to_numeric(df["PayloadMass"], errors="coerce")
    # EDA outputs
    fig, ax = plt.subplots(figsize=(9,5))
    df.groupby("LaunchSite")["Class"].mean().sort_values().plot(kind="bar", ax=ax)
    ax.set_title("First-stage landing success rate by launch site"); ax.set_ylabel("Success rate"); ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(OUT/"success_by_site.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    df.groupby("Year")["Class"].mean().plot(marker="o", ax=ax)
    ax.set_title("Landing success rate by year"); ax.set_ylabel("Success rate"); ax.set_ylim(0,1)
    fig.tight_layout(); fig.savefig(OUT/"success_by_year.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9,5))
    df.boxplot(column="PayloadMass", by="Class", ax=ax)
    ax.set_title("Payload mass by landing outcome"); ax.set_xlabel("Landing class (0=unsuccessful, 1=successful)")
    fig.suptitle(""); fig.tight_layout(); fig.savefig(OUT/"payload_by_class.png", dpi=160); plt.close(fig)

    # SQL-style results using SQLite through pandas; output query examples and results.
    import sqlite3
    db = sqlite3.connect(OUT/"spacex_analysis.sqlite")
    df.to_sql("launches", db, if_exists="replace", index=False)
    queries = {
      "success_by_site": "SELECT LaunchSite, COUNT(*) AS launches, ROUND(AVG(Class),3) AS success_rate FROM launches GROUP BY LaunchSite ORDER BY success_rate DESC",
      "success_by_orbit": "SELECT Orbit, COUNT(*) AS launches, ROUND(AVG(Class),3) AS success_rate FROM launches GROUP BY Orbit ORDER BY success_rate DESC",
      "yearly_trend": "SELECT substr(Date,1,4) AS year, COUNT(*) AS launches, ROUND(AVG(Class),3) AS success_rate FROM launches GROUP BY year ORDER BY year"
    }
    with open(OUT/"sql_results.txt","w") as f:
        for name,q in queries.items():
            f.write(f"## {name}\n{q}\n")
            f.write(pd.read_sql_query(q,db).to_string(index=False)+"\n\n")
    db.close()

    # Predictive analysis: exclude direct target and identifiers/free-text; encode categoricals.
    drop_cols = [c for c in ["Class","Date","Outcome","LandingPad","Serial","BoosterVersion"] if c in df.columns]
    X = df.drop(columns=drop_cols).copy()
    y = df["Class"].astype(int)
    X = pd.get_dummies(X, dummy_na=True)
    X = X.replace([float("inf"),float("-inf")], pd.NA).fillna(X.median(numeric_only=True)).fillna(0)
    # Guard against accidental all-null/constant columns.
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    models = {
      "Logistic Regression": Pipeline([("scale",StandardScaler()),("model",LogisticRegression(max_iter=3000,class_weight="balanced"))]),
      "SVM": Pipeline([("scale",StandardScaler()),("model",SVC(class_weight="balanced"))]),
      "Decision Tree": DecisionTreeClassifier(random_state=42,class_weight="balanced"),
      "KNN": Pipeline([("scale",StandardScaler()),("model",KNeighborsClassifier(n_neighbors=5))])
    }
    rows=[]; preds={}
    for name, model in models.items():
        model.fit(X_train,y_train); p=model.predict(X_test); preds[name]=p
        rows.append({"model":name,"accuracy":accuracy_score(y_test,p),"precision":precision_score(y_test,p,zero_division=0),
                     "recall":recall_score(y_test,p,zero_division=0),"f1":f1_score(y_test,p,zero_division=0)})
    metrics = pd.DataFrame(rows).sort_values("f1",ascending=False)
    metrics.to_csv(OUT/"model_metrics.csv",index=False)
    best_name=metrics.iloc[0]["model"]; best_pred=preds[best_name]
    pd.DataFrame(confusion_matrix(y_test,best_pred),index=["actual_0","actual_1"],columns=["predicted_0","predicted_1"]).to_csv(OUT/"confusion_matrix.csv")
    (OUT/"classification_report.txt").write_text(f"Best by F1 on this split: {best_name}\n\n"+classification_report(y_test,best_pred,zero_division=0))
    fig,ax=plt.subplots(figsize=(8,4.5))
    metrics.set_index("model")[["accuracy","precision","recall","f1"]].plot(kind="bar",ax=ax)
    ax.set_ylim(0,1); ax.set_title("Classification model comparison"); ax.set_ylabel("Score"); ax.tick_params(axis="x",rotation=20)
    fig.tight_layout(); fig.savefig(OUT/"model_comparison.png",dpi=160); plt.close(fig)
    print("Analysis complete. Check outputs/ for charts, SQL results, and measured model metrics.")
    print(metrics.to_string(index=False))

if __name__ == "__main__": main()
