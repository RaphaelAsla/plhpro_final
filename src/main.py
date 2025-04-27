from model import KNN
from plotter import Plotter
import pandas as pd

# ta paths einai relative, prepei na to trekseis enw eisai mesa sto src directory !!
train_data = pd.read_excel("./../data/Project40PastCampaignData.xlsx")
new_data = pd.read_excel("./../data/Project40NewCampaignData.xlsx")

predictor = KNN()
predictor.feed_data(train_data)
predictor.find_best_k()
predictor.fit()
predictions = predictor.predict(new_data)
predictor.make_metrics()
print(predictor.validation_metrics_str)
print(predictor.cv_validation_metrics)

#predictions = predictor.predict(
#    new_data, output_path="./../predictions/predictions.xlsx"
#)
#metrics = predictor.get_metrics()

#plotter = Plotter(metrics)
#
#fig1 = plotter.best_validation_metrics_vs_number_of_folds()
#fig1.savefig("./../plots/validation_metrics_vs_folds.png")
#
#fig2 = plotter.cv_precision_vs_k_value_for_different_folds()
#fig2.savefig("./../plots/cv_precision_vs_k.png")
#
#fig3 = plotter.cv_accuracy_vs_k_value_for_different_folds()
#fig3.savefig("./../plots/cv_accuracy_vs_k.png")
#
#fig4 = plotter.final_validation_metrics()
#fig4.savefig("./../plots/validation_results.png")
