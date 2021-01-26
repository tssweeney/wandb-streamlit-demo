from joblib import dump

import wandb
from sklearn import datasets, svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

artifact_name = "wandb-streamlit-example"
artifact_type = "model"
model_name="model.joblib"

# Load in data used by all models
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

def model_trainer(model_type="SVC"):
    # Start a run
    run = wandb.init(project="wandb-streamlit-demo", config={"model_type": model_type})

    # Create a classifier
    if model_type == "SVC":
        clf = svm.SVC(gamma=0.001)
    elif model_type == "LR":
        clf = LogisticRegression()
    else:
        clf = DummyClassifier()
    
    # Fit the classifier
    clf.fit(X_train, y_train)

    # Make predictions and log some metrics
    predicted = clf.predict(X_test)
    run.log(metrics.classification_report(y_test, predicted, output_dict=True))

    # Create an artifact and save the serialized model
    artifact = wandb.Artifact(artifact_name, artifact_type)
    with artifact.new_file(model_name, "wb") as file:
        dump(clf, file) 

    # Log the artifact
    run.log_artifact(artifact, aliases=[str(run.id)])

    # Log an iframe to view our streamlit app with URL params
    run.log({"model_viewer": wandb.Html("<iframe height='100%' width='100%' src='http://localhost:8501/?entity="+run.entity+"&project="+run.project+"&run_id="+run.id+"&artifact_name="+artifact_name+"&model_name="+model_name+"'></iframe>")})

    # Finish the run
    run.finish()

def main():
    # Here, we train 3 different models
    model_trainer("Dummy")
    model_trainer("LR")
    model_trainer("SVC")

main()
