# Melkor

Welcome to the Melkor project!

# How to run mlflow

## 1. Run  mlflow with `mlflow ui --backend-store-uri sqlite:///{DATABASE_NAME}.db`

## 2. Set the DATABASE_NAME as the tracking_uri when training and on inference

## 3. Run the ui or cli version and explore the nice graphs


# How to generate class diagrams

## 1. Install necessary packages
`pip install pylint, pip install graphviz, pip install pydot`


## 2. Generate .dot files
`pyreverse melkor`

## 3. Generate png out of .dot file
`dot -Tpng classes.dot > class_diagrams.png`
