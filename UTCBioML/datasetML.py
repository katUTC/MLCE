import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tkinter as tk
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay


class DSTreatment(object):

    # retrieve database location and initiate variables
    def __init__(self, file_loc, sheet, header, columns, feat, label, nant, neighbours, svc_kernel, svc_kernel_c,
                 svc_gamma, svc_gamma_c, gaussian_process, gaussian_process_rbf, decision_tree_max_depth,
                 random_forest_depth, random_forest_n_estimators, random_forest_max_feat, mlp_alpha, mlp_max_iter):
        self.file_loc = file_loc
        self.sheet = sheet
        self.header = header
        self.columns = columns
        self.feat = feat
        self.label = label
        self.nant = nant
        self.prep = None
        self.feats = None
        self.parameters = pd.DataFrame()
        self.analysis = []
        self.report_name = []
        self.report_score = []
        self.neighbours = neighbours
        self.svc_kernel = svc_kernel
        self.svc_kernel_c = svc_kernel_c
        self.svc_gamma = svc_gamma
        self.svc_gamma_c = svc_gamma_c
        self.gaussian_process = gaussian_process
        self.gaussian_process_rbf = gaussian_process_rbf
        self.decision_tree_max_depth = decision_tree_max_depth
        self.random_forest_depth = random_forest_depth
        self.random_forest_n_estimators = random_forest_n_estimators
        self.random_forest_max_feat = random_forest_max_feat
        self.mlp_alpha = mlp_alpha
        self.mlp_max_iter = mlp_max_iter

    plt.rcParams["figure.figsize"] = [18.00, 5.00]
    plt.rcParams["figure.autolayout"] = True

    # Set features and target/label based on columns and sheets
    def set_features(self, file_loc, sheet, header, columns, feat, label):
        df = pd.read_excel(file_loc, sheet_name=sheet, header=header, usecols=columns)
        df = df[[feat] + [c for c in df if c not in [feat, label]] + [label]]  # sets feat to beginning and label to end
        self.feats = df[df.columns.values]
        return self.feats

    # Analyse the dataset for the NaN values
    def count_nan(self, feats):
        prep = feats.copy(deep=True)
        length = len(prep.columns)
        nan_count = []

        for column in prep.iloc[:, 0:length]:
            nan_count.append(prep[column].isna().sum())

        table_data = [nan_count]
        rows1 = ['Count']
        headers = np.array(prep.columns.tolist())

        fig = plt.figure(figsize=(12, 4))
        fig.patch.set_visible(False)
        ax = plt.subplot(111)
        ax.axis('off')
        ax.axis('tight')
        the_table=ax.table(cellText=table_data, colLabels=headers, rowLabels=rows1, loc='center')
        ax.set_title('NaN values in database')
        the_table.auto_set_font_size(False)
        the_table.auto_set_column_width(col=list(range(len(headers))))
        the_table.set_fontsize(9)

        return plt.show()

    # Set the method by which NaN values are treated
    def set_nan_protocol(self, nant, feats, label):
        # Treatment of NaN or missing data points for deleting rows containing missing items
        if nant == 1:
            prep = feats.copy(deep=True)
            length = len(prep.columns)
            nan_before = []
            nan_after = []

            for column in prep.iloc[:, 0:length]:
                nan_before.append(prep[column].isna().sum())

            prep = feats.dropna().reset_index(drop=True)

            for column in prep.iloc[:, 0:length]:
                nan_after.append(prep[column].isna().sum())

            table_data = [nan_before, nan_after]
            rows1 = ['Before', 'After']
            headers = np.array(prep.columns.tolist())

            fig = plt.figure(figsize=(12, 6))
            fig.patch.set_visible(False)
            ax = fig.add_subplot(111)
            ax.axis('off')
            ax.axis('tight')
            the_table = ax.table(cellText=table_data, colLabels=headers, rowLabels=rows1, loc='center')
            ax.set_title('NaN value count before and after dropping NaN')
            the_table.auto_set_font_size(False)
            the_table.auto_set_column_width(col=list(range(len(headers))))
            the_table.set_fontsize(9)

            plt.show()

        elif nant == 2:
            # Treatment of NaN values by median replacement based on labels
            prep = feats.copy(deep=True)
            length = len(prep.columns)
            nan_before = []
            nan_after = []

            for column in prep.iloc[:, 0:length]:
                nan_before.append(prep[column].isna().sum())

            headers = np.array(prep.columns.tolist())
            means = prep.groupby(label).median()

            for column in prep.iloc[:, 0:length - 1]:
                prep[column] = prep.groupby(label).transform(lambda x: x.fillna(x.median()))[column]

            for column in prep.iloc[:, 0:length]:
                nan_after.append(prep[column].isna().sum())

            table_data = [nan_before, nan_after]
            rows1 = ['Before', 'After']

            fig = plt.figure()
            fig.patch.set_visible(False)
            ax = plt.subplot(211)
            ax.axis('off')
            ax.axis('tight')
            the_table = ax.table(cellText=table_data, colLabels=headers, rowLabels=rows1, loc='center')
            ax.set_title('NaN value count before and after median value replacement')
            the_table.auto_set_font_size(False)
            the_table.auto_set_column_width(col=list(range(len(headers))))
            the_table.set_fontsize(9)

            ax = plt.subplot(212)
            the_table2 = ax.table(cellText=means.values, colLabels=means.columns, rowLabels=means.index, loc='center')
            ax.axis('off')
            ax.axis('tight')
            ax.set_title('Label based median values used to replace NaN values')
            the_table2.auto_set_font_size(False)
            the_table2.auto_set_column_width(col=list(range(len(means.columns))))
            the_table2.set_fontsize(9)

            plt.show()
        elif nant == 3:
            # Treatment of NaN values by mode replacement based on labels
            prep = feats.copy(deep=True)
            length = len(prep.columns)
            nan_before = []
            nan_after = []

            for column in prep.iloc[:, 0:length]:
                nan_before.append(prep[column].isna().sum())

            headers = np.array(prep.columns.tolist())

            for column in prep.iloc[:, 0:length - 1]:
                f = lambda x: x.median() if np.issubdtype(x.dtype, np.number) else x.mode().iloc[column]
                prep = prep.fillna(prep.groupby(label).transform(f))
                modes = prep.groupby(label).agg(f)

            for column in prep.iloc[:, 0:length]:
                nan_after.append(prep[column].isna().sum())

            table_data = [nan_before, nan_after]
            rows1 = ['Before', 'After']

            fig = plt.figure()
            fig.patch.set_visible(False)
            ax = plt.subplot(211)
            ax.axis('off')
            ax.axis('tight')
            the_table = ax.table(cellText=table_data, colLabels=headers, rowLabels=rows1, loc='center')
            ax.set_title('NaN value count before and after median / mode value replacement')
            the_table.auto_set_font_size(False)
            the_table.auto_set_column_width(col=list(range(len(headers))))
            the_table.set_fontsize(9)

            ax = plt.subplot(212)
            the_table2 = ax.table(cellText=modes.values, colLabels=modes.columns, rowLabels=modes.index, loc='center')
            ax.axis('off')
            ax.axis('tight')
            ax.set_title('Label based mode values used to replace NaN values')
            the_table2.auto_set_font_size(False)
            the_table2.auto_set_column_width(col=list(range(len(modes.columns))))
            the_table2.set_fontsize(9)

            plt.show()
        else:
            # Treatment of NaN values by mean replacement based on labels
            prep = feats.copy(deep=True)
            length = len(prep.columns)
            nan_before = []
            nan_after = []

            for column in prep.iloc[:, 0:length]:
                nan_before.append(prep[column].isna().sum())

            headers = np.array(prep.columns.tolist())
            means = prep.groupby(label).mean()

            for column in prep.iloc[:, 0:length-1]:
                prep[column] = prep.groupby(label).transform(lambda x: x.fillna(x.mean()))[column]

            for column in prep.iloc[:, 0:length]:
                nan_after.append(prep[column].isna().sum())

            table_data = [nan_before, nan_after]
            rows1 = ['Before', 'After']

            fig = plt.figure()
            fig.patch.set_visible(False)
            ax = plt.subplot(211)
            ax.axis('off')
            ax.axis('tight')
            the_table = ax.table(cellText=table_data, colLabels=headers, rowLabels=rows1, loc='center')
            ax.set_title('NaN value count before and after mean value replacement')
            the_table.auto_set_font_size(False)
            the_table.auto_set_column_width(col=list(range(len(headers))))
            the_table.set_fontsize(9)

            ax = plt.subplot(212)
            the_table2 = ax.table(cellText=means.values, colLabels=means.columns, rowLabels=means.index, loc='center')
            ax.axis('off')
            ax.axis('tight')
            ax.set_title('Label based mean values used to replace NaN values')
            the_table2.auto_set_font_size(False)
            the_table2.auto_set_column_width(col=list(range(len(means.columns))))
            the_table2.set_fontsize(9)

            # plt.show()
            plt.tight_layout
            plt.show()

        self.prep = prep
        return self.prep

    def launch_ml(self, prep, nant, neighbours, svc_kernel, svc_kernel_c, svc_gamma, svc_gamma_c, gaussian_process,
                  gaussian_process_rbf, decision_tree_max_depth, random_forest_depth, random_forest_n_estimators,
                  random_forest_max_feat, mlp_alpha, mlp_max_iter):
        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]

        classifiers = [
            KNeighborsClassifier(neighbours),
            SVC(kernel=svc_kernel, C=svc_kernel_c),
            SVC(gamma=svc_gamma, C=svc_gamma_c),
            GaussianProcessClassifier(gaussian_process * RBF(gaussian_process_rbf)),
            DecisionTreeClassifier(max_depth=decision_tree_max_depth),
            RandomForestClassifier(max_depth=random_forest_depth, n_estimators=random_forest_n_estimators,
                                   max_features=random_forest_max_feat),
            MLPClassifier(alpha=mlp_alpha, max_iter=mlp_max_iter),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis(),
        ]

        length = len(prep.columns)

        frames = pd.DataFrame(
            [prep.iloc[:, 0], prep.iloc[:, l + 1], prep.iloc[:, -1]] for l in range(length-2)
            )

        headers = np.array(prep.columns.tolist())
        rows = len(frames)

        datasets = []
        for i in range(rows):
            datasets.append([np.transpose(np.array([frames.iloc[i, 0].values, frames.iloc[i, 1].values])),
                             np.transpose(np.array(frames.iloc[i, -1].values))])

        # print("this is the dataset:", datasets)
        if nant == 0:
            treatment = '(NaN values replaced with means)'
        elif nant == 1:
            treatment = '(NaN values deleted)'
        elif nant == 2:
            treatment = '(NaN values replaced with median)'
        elif nant == 3:
            treatment = '(NaN values replaced with mean if numerical or mode if categorical)'

        title = 'For target: ' + str(headers[-1]) + ' , ' + str(headers[0]) + \
                ' analysis for the following respective features:\n' + \
                str(headers[1:-1]).replace('[', '').replace(']', '') + '\n' + str(treatment)

        figure = plt.figure(figsize=(27, 9))
        figure.suptitle(str(title), fontsize=16)
        i = 1

        # iterate over datasets
        for ds_cnt, ds in enumerate(datasets):
            # preprocess dataset, split into training and test part
            X, y = ds
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.4
            )

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

            if x_min is None:
                x_min = np.nanmin(X[:, 0]) - 1
            if x_max is None:
                x_max = np.nanmax(X[:, 0]) + 1
            if y_min is None:
                y_min = np.nanmin(X[:, 1]) - 1
            if y_max is None:
                y_max = np.nanmax(X[:, 1]) + 1

            # just plot the dataset first
            cm = plt.cm.RdBu
            cm_bright = ListedColormap(["#FF0000", "#0000FF"])
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            if ds_cnt == 0:
                ax.set_title("Input data")
            # Plot the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
            # Plot the testing points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k"
            )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())

            i += 1

            # iterate over classifiers
            for name, clf in zip(names, classifiers):
                ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

                clf = make_pipeline(StandardScaler(), clf)
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                DecisionBoundaryDisplay.from_estimator(
                    clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

                aa = {clf: clf.get_params()}
                dfp = pd.DataFrame.from_dict(aa, orient='index')
                dfp = dfp.transpose()
                self.parameters = pd.concat([self.parameters, dfp])
                self.report_name.append(name)
                self.report_score.append(score)

                # Plot the training points
                ax.scatter(
                    X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k"
                )
                # Plot the testing points
                ax.scatter(
                    X_test[:, 0],
                    X_test[:, 1],
                    c=y_test,
                    cmap=cm_bright,
                    edgecolors="k",
                    alpha=0.6,
                )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks(())
                ax.set_yticks(())

                if ds_cnt == 0:
                    ax.set_title(name)
                ax.text(
                    x_max - 0.3,
                    y_min + 0.3,
                    ("%.2f" % score).lstrip("0"),
                    size=15,
                    horizontalalignment="right",
                )
                i += 1

        plt.tight_layout()

        figure = plt.show()

        #return plt.show()

        # Excel printout of parameters
        '''location = tk.Tk()
        location.withdraw()
        file_loc2 = tk.simpledialog.askstring(title="Excel_save_to", prompt="Enter path+name.xlsx:")
        writer = pd.ExcelWriter(file_loc2)
        self.parameters.to_excel(writer, sheet_name='Parameters')

        for column in self.parameters:
            column_length = max(self.parameters[column].astype(str).map(len).max(), len(column))
            col_idx = self.parameters.columns.get_loc(column)
            writer.sheets['Parameters'].set_column(col_idx, col_idx, column_length)

        writer.close()'''

        repeats = len(datasets)
        '''for r in range(repeats):
            for n in range(10):
                # feature = 'feature' + str(r)
                feature = str(headers[1:-1])
                self.analysis.append(feature)'''
        for r in range(repeats):
            for n in range(10):
                feature = str(headers[r+1])
                self.analysis.append(feature)
        df = pd.DataFrame(list(zip(self.analysis, self.report_name, self.report_score)), columns=['Analysis', 'ML Algorithm', 'Score'])
        df = df.pivot(index='ML Algorithm', columns='Analysis', values='Score')
        fig = plt.figure(figsize=(18, 6))
        fig.patch.set_visible(False)
        ax = plt.subplot(111)
        ax.axis('off')
        ax.axis('tight')
        the_table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
        ax.set_title('Score for each classification model')
        the_table.auto_set_font_size(False)
        the_table.auto_set_column_width(col=list(range(len(df.columns))))
        the_table.set_fontsize(9)

        figure2 = plt.show()

        return figure, figure2

