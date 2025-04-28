import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Plotter:
    def __init__(self, metrics):
        self.cv_validation_metrics = metrics["cv_validation_metrics"]
    # tha to ksana grapsw, htan poly messy
