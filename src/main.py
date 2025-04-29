from model import KNN
from plotter import Plotter
import pandas as pd

# auta tha ginontai mesa sto GUI apo ton xrhsth 
########################################################################################
# (ta paths einai relative, prepei na to trekseis enw eisai mesa sto src directory) !! #
train_data = pd.read_excel("./../data/Project40PastCampaignData.xlsx")                 #
new_data = pd.read_excel("./../data/Project40NewCampaignData.xlsx")                    #
########################################################################################

if __name__ == "__main__":
    # auta epishs tha ginontai mesa sto GUI, o xrhsths tha mporei na kanei define monos toy ton aritho twn neighbors h an thelei na ginei aytomata to pick (find_best_neighbors)
    ########################################################################################################################################################################################################################
    knn = KNN() # <- neighbors go here (eg KNN(3))
    knn.feed_data(train_data) # <- pretty self explanatory

    optimize_on = "accuracy" # / "precision"
    # mporoyme na valoyme kai custom ranges gia neighbors (default 2 - 21) kai folds (2, 11), des (line 30 model.py) (pairnei ligh wra, an thes na testareis kalytera kanto me pre defined neighbors (eg KNN(3))
    knn.find_best_neighbors(optimize_on, range(2, 21), range(2, 11)) # <- AN den exoyme thesei monoi mas tous neighbors (px predictor = KNN(3)), trexoyme ayto me stoxo na vroyme to poio optimal (pio balanced synhthws, alla whatever) plithos gia tous neighbors

    knn.fit() # <- training

    predictions = knn.predict(new_data) # <- pretty self explanatory (returns pandas DataFrame)
    #predictions = knn.predict(new_data, put_path="./../predictions/predictions.xlsx") # <- mporoyme na ta kanoyme kai kateythian save

    knn.gen_metrics() # <- kanei generate ola ta metrics

    # an tha thes se dict ->  knn.validation_metrics (line 176 model.py gia na deis structure) 
    print(knn.validation_metrics_str) # <- ta idia metrics alla se ENA string an aplos thes na ta valeis se kapoio pedio sto gui
    ########################################################################################################################################################################################################################

    # Testing the plotter (ignore)
    ##################################################################
    plotter = Plotter(knn.overall_validation_metrics)

    # an kaneis specify path kanoyn mono save, an den kaneis, ta vlepeis on run
    #plotter.plot_neighbors_vs_metric_per_fold(optimize_on, "./../plots/plot_neighbors_vs_metric_per_fold")
    #plotter.plot_mean_metric_per_fold(optimize_on, "./../plots/plot_mean_metric_per_fold")
    plotter.plot_trisurf_metric_per_fold(optimize_on, "./../plots/plot_plot_trisurf_metric_per_fold")
    #################################################################

    # ta exw edw gia testing, sto telos, mono ayta prepei na vriskontai edw
    #gui = GUI(width, height, title)
    #gui.run()
