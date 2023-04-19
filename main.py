import tkinter as tk
import UTCBioML as utc
import pandas
import matplotlib.pyplot as plt

from tkinter import simpledialog
from tkinter import ttk
from tkinter import *
from PIL import ImageTk
from PIL import Image
from UTCBioML import datasetML


class Application:
    def __init__(self, master):
        # tk.Frame.__init__(self, master)
        self.master = master
        # Set up for all the data entry windows linking to the excel sheet
        self.feat1 = tk.StringVar()
        self.feat2 = tk.StringVar()
        self.feat3 = tk.StringVar()
        self.nant = 0
        self.file_loc = None
        self.sheet = None
        self.columns = None
        self.header = None
        self.feat = None
        self.prep = None
        self.label = None
        self.file_loc = None

        self.aa = tk.Label(master, text="Machine learning classification", font="5").place(x=80, y=0)
        self.bb = tk.Label(master, text="for Excel spreadsheets", font="5").place(x=110, y=25)
        self.file_path_name = tk.Label(master, text="Excel file path:").place(x=30, y=70)
        self.sheet_name = tk.Label(master, text="Source sheet name:").place(x=30, y=100)
        self.header_name = tk.Label(master, text="Header size to ignore:").place(x=30, y=130)
        self.columns_name = tk.Label(master, text="Use columns (ex: C,D,E,F):").place(x=30, y=180)
        self.feat_name = tk.Label(master, text="Name of feature to compare:").place(x=30, y=210)
        self.label_name = tk.Label(master, text="Label/target feature:").place(x=30, y=240)
        self.file_loc_entry = tk.Entry(master)
        self.file_loc_entry.place(x=210, y=70)
        self.sheet_entry = tk.Entry(master)
        self.sheet_entry.place(x=210, y=100)
        self.header_entry = tk.Entry(master)
        self.header_entry.place(x=210, y=130)
        self.columns_entry = tk.Entry(master)
        self.columns_entry.place(x=210, y=180)
        self.feat1_entry = tk.Entry(master)
        self.feat1_entry.place(x=210, y=210)
        self.label_entry = tk.Entry(master)
        self.label_entry.place(x=210, y=240)

        # Hyper parameter initiation
        self.neighbours = 3
        self.svc_kernel = "linear"
        self.svc_kernel_c = 0.025
        self.svc_gamma = 2
        self.svc_gamma_c = 1
        self.gaussian_process = 1.0
        self.gaussian_process_rbf = 1.0
        self.decision_tree_max_depth = 5
        self.random_forest_depth = 5
        self.random_forest_n_estimators = 10
        self.random_forest_max_feat = 1
        self.mlp_alpha = 1
        self.mlp_max_iter = 1000

        plt.rcParams["figure.figsize"] = [9.00, 5.00]
        plt.rcParams["figure.autolayout"] = True

        # Dropdown menu options for analysis
        self.options = [
            "Run_Missing_Data_Check",
            "Set_ML_Hyper-Parameters",
            "Run_Machine_Learning_Classification"
        ]

        # datatype of menu text for analysis
        self.clicked = StringVar(master)
        # datatype of menu text for Nan treatment
        self.nan_treat = StringVar(master)

        # initial menu text for analysis to run
        self.clicked.set("Select Operation")  # default value

        # Create Label for analysis
        self.pre_label = Label(master, text="Current selection:")
        self.pre_label.place(x=30, y=320)
        self.label1 = Label(master, text=self.clicked.get())
        self.label1.place(x=50, y=350)

        # Create Dropdown menu for analysis
        self.drop = OptionMenu(master, self.clicked, *self.options)
        self.clicked.trace('w', self.show)
        self.drop.place(x=30, y=290)

        # Dropdown menu options for NaN Treatment
        self.options2 = [
            "Delete NaN values (recommended)",
            "Replace NaN values with mean",
            "Replace NaN values with median",
            "Replace NaN w/ mean (num) or mode (cat)",
        ]

        # initial menu text for Nan Treatment
        self.nan_treat.set("Select NaN Treatment for ML")  # default value

        # Create Label for NaN treatment
        self.pre_label2 = Label(master, text="Current selection:")
        self.pre_label2.place(x=30, y=430)
        self.label2 = Label(master, text=self.nan_treat.get())
        self.label2.place(x=50, y=460)

        # Create Dropdown menu for Nan Treatment
        self.drop2 = OptionMenu(master, self.nan_treat, *self.options2)
        self.nan_treat.trace('w', self.show2)
        self.drop2.place(x=30, y=400)

        # Create button, it will launch the operation
        self.button = Button(master, text="Launch", command=self.launch)
        self.button.place(x=160, y=510)

        # Creation of the logos of our partners
        self.img_utc = Image.open("UTCBioML/utc.png")
        self.img_utc = self.img_utc.resize((120, 30), Image.Resampling.LANCZOS)
        self.img_utc = ImageTk.PhotoImage(self.img_utc)
        self.utc = Label(master, image=self.img_utc)
        self.utc.place(x=10, y=570)

        self.img_investir = Image.open("UTCBioML/investir.png")
        self.img_investir = self.img_investir.resize((30, 30), Image.Resampling.LANCZOS)
        self.img_investir = ImageTk.PhotoImage(self.img_investir)
        self.investir = Label(master, image=self.img_investir)
        self.investir.place(x=140, y=570)

        self.img_region = Image.open("UTCBioML/region.png")
        self.img_region = self.img_region.resize((100, 20), Image.Resampling.LANCZOS)
        self.img_region = ImageTk.PhotoImage(self.img_region)
        self.region = Label(master, image=self.img_region)
        self.region.place(x=190, y=580)

        self.img_labex = Image.open("UTCBioML/labex.png")
        self.img_labex = self.img_labex.resize((60, 40), Image.Resampling.LANCZOS)
        self.img_labex = ImageTk.PhotoImage(self.img_labex)
        self.labex = Label(master, image=self.img_labex)
        self.labex.place(x=310, y=560)

    # Change the label text for the selected analysis
    def show(self, *args):
        self.label1.config(text=self.clicked.get())

    # Change the label text for the NaN treatment option
    def show2(self, *args):
        self.label2.config(text=self.nan_treat.get())

    def launch(self, *args):
        if self.clicked.get() == "Run_Missing_Data_Check":
            print("Selected analysis is 1:", self.clicked.get())
            self.file_loc = str(self.file_loc_entry.get())
            self.sheet = str(self.sheet_entry.get())
            self.header = int(self.header_entry.get())
            self.columns = str(self.columns_entry.get())
            self.feat = str(self.feat1_entry.get())
            self.label = str(self.label_entry.get())
            features = utc.datasetML.DSTreatment(self.file_loc, self.sheet, self.header, self.columns,
                                                  self.feat, self.label, self.nant, self.neighbours, self.svc_kernel,
                                                  self.svc_kernel_c, self.svc_gamma, self.svc_gamma_c,
                                                  self.gaussian_process, self.gaussian_process_rbf,
                                                  self.decision_tree_max_depth, self.random_forest_depth,
                                                  self.random_forest_n_estimators, self.random_forest_max_feat,
                                                  self.mlp_alpha, self.mlp_max_iter)
            feats = features.set_features(self.file_loc, self.sheet, self.header, self.columns, self.feat,
                                          self.label)
            features.count_nan(feats)
        elif self.clicked.get() == "Set_ML_Hyper-Parameters":
            print("Selected analysis is 2:", self.clicked.get())
            self.hyper(self)
        elif self.clicked.get() == "Run_Machine_Learning_Classification":
            print("Selected analysis is 3:", self.clicked.get())
            #print("This is get_nan:", self.nan_treat.get())
            if self.nan_treat.get() == "Delete NaN values (recommended)":
                self.nant = 1
                self.run_nan(self)
            elif self.nan_treat.get() == "Replace NaN values with mean":
                self.nant = 0
                self.run_nan(self)
            elif self.nan_treat.get() == "Replace NaN values with median":
                self.nant = 2
                self.run_nan(self)
            elif self.nan_treat.get() == "Replace NaN w/ mean (num) or mode (cat)":
                self.nant = 3
                self.run_nan(self)
            else:
                print("Whoops ! We've got a problem !")

        else:
            print("Whoops ! First select how to treat NaN (or missing) values !")

    def run_nan(self, *args):
        self.file_loc = str(self.file_loc_entry.get())
        self.sheet = str(self.sheet_entry.get())
        self.header = int(self.header_entry.get())
        self.columns = str(self.columns_entry.get())
        self.feat = str(self.feat1_entry.get())
        self.label = str(self.label_entry.get())
        features = utc.datasetML.DSTreatment(self.file_loc, self.sheet, self.header, self.columns, self.feat,
                                              self.label, self.nant, self.neighbours, self.svc_kernel,
                                              self.svc_kernel_c, self.svc_gamma, self.svc_gamma_c,
                                              self.gaussian_process, self.gaussian_process_rbf,
                                              self.decision_tree_max_depth, self.random_forest_depth,
                                              self.random_forest_n_estimators, self.random_forest_max_feat,
                                              self.mlp_alpha, self.mlp_max_iter)
        feats = features.set_features(self.file_loc, self.sheet, self.header, self.columns, self.feat, self.label)
        self.prep = features.set_nan_protocol(self.nant, feats, self.label)
        features.launch_ml(self.prep, self.nant, self.neighbours, self.svc_kernel, self.svc_kernel_c, self.svc_gamma,
                           self.svc_gamma_c, self.gaussian_process, self.gaussian_process_rbf,
                           self.decision_tree_max_depth, self.random_forest_depth, self.random_forest_n_estimators,
                           self.random_forest_max_feat, self.mlp_alpha, self.mlp_max_iter)

    def hyper(self, *args):
        self.hyper_import = tk.Tk()
        self.hyper_import.title("Hyper parameter set-up")
        self.hyper_import.geometry('430x520')
        self.hyper_name = tk.Label(self.hyper_import, text="Hyper-parameter initiation:").place(x=120, y=10)
        self.neighbours_name = tk.Label(self.hyper_import, text="KNearest Neighbours (initial=3):").place(x=30, y=50)
        self.neighbours_entry = tk.Entry(self.hyper_import)
        self.neighbours_entry.insert(0, "3")
        self.neighbours_entry.place(x=270, y=50)
        self.svc_kernel_name = tk.Label(self.hyper_import, text="SVC Kernel (initial= linear):").place(x=30, y=80)
        self.svc_kernel_entry = tk.Entry(self.hyper_import)
        self.svc_kernel_entry.insert(0, "linear")
        self.svc_kernel_entry.place(x=270, y=80)
        self.svc_kernel_c_name = tk.Label(self.hyper_import, text="SVC Kernel C value (initial=0.025):").place(x=30, y=110)
        self.svc_kernel_c_entry = tk.Entry(self.hyper_import)
        self.svc_kernel_c_entry.insert(0, "0.025")
        self.svc_kernel_c_entry.place(x=270, y=110)
        self.svc_gamma_name = tk.Label(self.hyper_import, text="SVC Gamma (initial=2):").place(x=30, y=140)
        self.svc_gamma_entry = tk.Entry(self.hyper_import)
        self.svc_gamma_entry.insert(0, "2")
        self.svc_gamma_entry.place(x=270, y=140)
        self.svc_gamma_c_name = tk.Label(self.hyper_import, text="SVC Gamma C value (initial=1):").place(x=30, y=170)
        self.svc_gamma_c_entry = tk.Entry(self.hyper_import)
        self.svc_gamma_c_entry.insert(0, "1")
        self.svc_gamma_c_entry.place(x=270, y=170)
        self.gaussian_process_name = tk.Label(self.hyper_import, text="Gaussian Process (initial=1.0):").place(x=30, y=200)
        self.gaussian_process_entry = tk.Entry(self.hyper_import)
        self.gaussian_process_entry.insert(0, "1.0")
        self.gaussian_process_entry.place(x=270, y=200)
        self.gaussian_process_rbf_name = tk.Label(self.hyper_import, text="Gaussian Process RBF (initial=1.0):").place(x=30, y=230)
        self.gaussian_process_rbf_entry = tk.Entry(self.hyper_import)
        self.gaussian_process_rbf_entry.insert(0, "1.0")
        self.gaussian_process_rbf_entry.place(x=270, y=230)
        self.decision_tree_name = tk.Label(self.hyper_import, text="Decision Tree max depth (initial=5):").place(x=30, y=260)
        self.decision_tree_entry = tk.Entry(self.hyper_import)
        self.decision_tree_entry.insert(0, "5")
        self.decision_tree_entry.place(x=270, y=260)
        self.random_forest_depth_name = tk.Label(self.hyper_import, text="Random forest depth (initial=5):").place(x=30, y=290)
        self.random_forest_depth_entry = tk.Entry(self.hyper_import)
        self.random_forest_depth_entry.insert(0, "5")
        self.random_forest_depth_entry.place(x=270, y=290)
        self.random_forest_n_name = tk.Label(self.hyper_import, text="Random forest N estimators (initial=10):").place(x=30, y=320)
        self.random_forest_n_entry = tk.Entry(self.hyper_import)
        self.random_forest_n_entry.insert(0, "10")
        self.random_forest_n_entry.place(x=270, y=320)
        self.random_forest_max_feat_name = tk.Label(self.hyper_import, text="Random forest max features (initial=1):").place(x=30, y=350)
        self.random_forest_max_feat_entry = tk.Entry(self.hyper_import)
        self.random_forest_max_feat_entry.insert(0, "1")
        self.random_forest_max_feat_entry.place(x=270, y=350)
        self.mlp_alpha_name = tk.Label(self.hyper_import, text="MLP alpha (initial=1):").place(x=30, y=380)
        self.mlp_alpha_entry = tk.Entry(self.hyper_import)
        self.mlp_alpha_entry.insert(0, "1")
        self.mlp_alpha_entry.place(x=270, y=380)
        self.mlp_max_iter_name = tk.Label(self.hyper_import, text="MLP maximum iterations (initial=1000):").place(x=30, y=410)
        self.mlp_max_iter_entry = tk.Entry(self.hyper_import)
        self.mlp_max_iter_entry.insert(0, "1000")
        self.mlp_max_iter_entry.place(x=270, y=410)
        self.button2 = Button(self.hyper_import, text="Validate", command=self.set_hyper)
        self.button2.place(x=170, y=460)

    def set_hyper(self):
        self.neighbours = int(self.neighbours_entry.get())
        self.svc_kernel = str(self.svc_kernel_entry.get())
        self.svc_kernel_c = float(self.svc_kernel_c_entry.get())
        self.svc_gamma = int(self.svc_gamma_entry.get())
        self.svc_gamma_c = float(self.svc_gamma_c_entry.get())
        self.gaussian_process = float(self.gaussian_process_entry.get())
        self.gaussian_process_rbf = float(self.gaussian_process_rbf_entry.get())
        self.decision_tree_max_depth = int(self.decision_tree_entry.get())
        self.random_forest_depth = int(self.random_forest_depth_entry.get())
        self.random_forest_n_estimators = int(self.random_forest_n_entry.get())
        self.random_forest_max_feat = int(self.random_forest_max_feat_entry.get())
        self.mlp_alpha = int(self.mlp_alpha_entry.get())
        self.mlp_max_iter = int(self.mlp_max_iter_entry.get())
        self.hyper_import.withdraw()

        table_data = pandas.DataFrame({'Hyperparameters': ['KNN', 'SVC Kernel Type', 'SVC Kernel C Value', 'SVC Gamma',
                                                           'SVC Gamma C Value', 'Gaussian Process',
                                                           'Gaussian Process RBF', 'Decision Tree Max Depth',
                                                           'Random Forest Depth', 'Random Forest N Estimators',
                                                           'Random Forest Maximum Features', 'MLP Alpha',
                                                           'MLP Maximum Iterations'],
                                       'Values': [self.neighbours, self.svc_kernel, self.svc_kernel_c, self.svc_gamma,
                                                  self.svc_gamma_c, self.gaussian_process, self.gaussian_process_rbf,
                                                  self.decision_tree_max_depth, self.random_forest_depth,
                                                  self.random_forest_n_estimators, self.random_forest_max_feat,
                                                  self.mlp_alpha, self.mlp_max_iter]
                                       })

        fig = plt.figure()
        fig.patch.set_visible(False)
        ax = plt.subplot(111)
        ax.axis('off')
        ax.axis('tight')
        the_table = ax.table(cellText=table_data.values, colLabels=table_data.columns, loc='center')
        ax.set_title('Machine learning hyper parameter report')
        the_table.auto_set_font_size(False)
        the_table.auto_set_column_width(col=list(range(len(table_data.columns))))
        the_table.set_fontsize(9)

        plt.show()


window = tk.Tk()
window.title("MLCE by BMBI")
window.geometry('380x620')  # width by height
app = Application(window)
window.mainloop()
