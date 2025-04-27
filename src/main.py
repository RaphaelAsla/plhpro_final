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
predictions = predictor.predict(
    new_data, output_path="./../predictions/predictions.xlsx"
)
predictor.gen_metrics()
print(predictor.validation_metrics_str)


plotter = Plotter(predictor.overall_validation_metrics)

fig1 = plotter.plot_best_cv_metrics_vs_folds()
if fig1 is not None:
    fig1.savefig("./../plots/plot_best_cv_metrics_vs_folds.png")

fig2 = plotter.plot_cv_metrics_vs_k("precision")
if fig2 is not None:
    fig2.savefig("./../plots/plot_cv_metrics_vs_k (precision).png")

fig3 = plotter.plot_cv_metrics_vs_k("accuracy")
if fig3 is not None:
    fig3.savefig("./../plots/plot_cv_metrics_vs_k (accuracy).png")

fig4 = plotter.plot_cv_metrics_heatmap('precision')
if fig4 is not None:
    fig4.savefig("./../plots/plot_cv_metrics_heatmap (precision).png")

fig5 = plotter.plot_cv_metrics_heatmap('accuracy')
if fig5 is not None:
    fig5.savefig("./../plots/plot_cv_metrics_heatmap (accuracy).png")

fig6 = plotter.plot_test_validation_metrics()
fig6.savefig("./../plots/plot_test_validation_metrics.png")

fig7 = plotter.plot_confusion_matrix()
fig7.savefig("./../plots/plot_confusion_matrix.png")

