class Strategy:
    def __init__(self, training_dataset_filepath, predict_dataset_filepath):
        self.training_data = training_dataset_filepath
        self.predict_data = predict_dataset_filepath
        self.signals = []
        
    def generate_signals(self):
        raise NotImplementedError("This method should be overridden by subclasses")
