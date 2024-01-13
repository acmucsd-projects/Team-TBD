"""
ACM AI Project Team: TBD
This file contains utility functions needed to train the model.
"""

def model_train(model, train_dataset, val_dataset, epoch=5):
    hist = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epoch
    )
    return hist

def save_model(model, path):
    model.save(path)