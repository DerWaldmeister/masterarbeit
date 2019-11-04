import tflearn
class MonitorCallback(tflearn.callbacks.Callback):
    def __init__(self):
        self.accuracies = []
        self.trainLosses = []

    def on_epoch_end(self, training_state):
        self.accuracies.append(training_state.global_acc)
        self.trainLosses.append(training_state.global_loss)


