from joblib import load

import wandb
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

scale = 20

# Downloads the model artifact from wandb
@st.cache
def get_model(entity, project, run_id, artifact_name, model_name):
    api = wandb.PublicApi()
    run = api.run("{}/{}/{}".format(entity, project, run_id))
    artifact = api.artifact("{}/{}/{}:{}".format(entity, project, artifact_name, run_id))
    model_path = artifact.get_path(model_name).download()
    model = load(model_path)
    return model

# Makes a canvas that the user can draw on to test the model
def make_canvas(on_data):
    canvas_result = st_canvas(
        fill_color="rgb(0, 0, 0)",
        stroke_width=scale * 1.5,
        stroke_color="rgb(255, 255, 255)",
        background_color="rgb(0, 0, 0)",
        update_streamlit=True,
        width=8*scale,
        height=8*scale,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        on_data(canvas_result.image_data)

def main():
    # Extract params from URL
    query_params = st.experimental_get_query_params()
    entity = query_params.get("entity",[None])[0]
    project = query_params.get("project",[None])[0]
    run_id = query_params.get("run_id",[None])[0]
    artifact_name = query_params.get("artifact_name",[None])[0]
    model_name = query_params.get("model_name",[None])[0]
    if entity is None or project is None or run_id is None or artifact_name is None or model_name is None:
        st.write("Invalid URL Params, got: " + str(query_params))
    else:
        # Get the model
        model = get_model(entity, project, run_id, artifact_name, model_name)
        if model:
            # When the user draws a figure, transform the data into something the model
            # can intake and perform a prediction
            def handle_data(data):
                _data = np.zeros((8,8))
                data = np.array(data)[:,:,0]
                for i in range(8):
                    for j in range(8):
                        _data[i,j] = (data[i*scale:(i+1)*scale,j*scale:(j+1)*scale].mean() / 16)
                _data = _data.astype(int)
                st.header("Prediction: " + str(model.predict(_data.reshape(1, -1))[0]))
            make_canvas(handle_data)
        else:
            st.write("Model not found")

main()
