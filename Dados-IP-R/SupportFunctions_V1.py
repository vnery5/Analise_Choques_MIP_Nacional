### ============================================================================================
### Passagem do R-Script desenvolvido por VALE, PEROBELLI para o Python
### ============================================================================================

## Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

### ============================================================================================

def read_matrices(sFile, sSheet):
    """
    Reads the Excel spreadsheets and converts them to Numpy arrays
    :param sFile: Name of the Excel File
    :param sSheet: Name of the spreadsheet
    :return: Numpy array of the desired sheet
    """

    ## Reading the Excel file using pandas
    df = pd.read_excel(sFile, sheet_name=sSheet, header=None)

    ## Convert it to a numpy array
    df = df.to_numpy()

    return df

def bar_plot(vData, vXLabels, sTitle, sXTitle, sFigName, sCaption="",
             yadjust=0.001, saveFig=False, sBarColor="#595959", bAnnotate=True):
    """
    Creates a styled bar plot containing the data
    :param vData: vector (1D array) containing the data to be plotted
    :param vXLabels: vector containing the x axis labels
    :param sTitle: string containing the title
    :param sXTitle: string containg the title for the x axis
    :param sCaption: string containing the title. Defaults to nothing
    :param yadjust: float that adjusts the height of the annotations. Defaults to 0.001.
    :param sFigName: desired file name of the saved figure (without the extension).
        The figures are saved in the "Figuras" subdirectory.
    :param saveFig: boolean; whether to save graph or not in the 'Figuras' subdirectory
    :param sBarColor: color (string) to fill the bars. Defaults to gray.
    :param bAnnotate: whether to annotate bars or not. Defaults to True.
    :return:
        fig: matplotlib object
        Also, saves the plot (in pdf) to the "Figuras" subdirectory.
    """
    if not saveFig:
        return 0
    else:
        # Creating fig object
        fig, ax = plt.subplots(figsize=(12, 8))

        # Creating grid and gray bars
        plt.bar(x=vXLabels, height=vData, color=sBarColor, zorder=3)
        ax.grid(zorder=0)
        # Painting the outside of the plot with the color light blue
        fig.set_facecolor("#e6f2ff")

        # Setting title and adjusting axis
        ax.set_title(sTitle, fontsize=18)
        ax.set_xlabel(sXTitle, fontsize=13)
        plt.tick_params(axis='x', which='major', labelsize=12)
        plt.tick_params(axis='y', which='major', labelsize=12)

        # Creating caption
        plt.figtext(0.125, 0.005, sCaption,
                    wrap=True, horizontalalignment='left', fontsize=12)

        # Annotating the bars (if requested)
        if bAnnotate:
            for patch, label in zip(ax.patches, vData):
                height = patch.get_height()
                ax.text(
                    patch.get_x() + patch.get_width() / 2, height + yadjust,
                    np.around(label, 2), ha="center", va="bottom", fontsize=12
                )

        # fig.tight_layout()
        ## Saving the figure
        sFileName = "Figuras/" + sFigName + ".pdf"
        # fig.show()
        fig.savefig(sFileName, dpi=1200)

        return fig

def named_scatter_plot(x, y, inf_lim, sup_lim, sTitle, vLabels, sXTitle,
                       sYTitle, sFigName, sCaption="", saveFig=False, nTextLimit=0.045):
    """
    Creates a styled scatterplot containing the data and labels
    :param x: data for the x axis
    :param y: data for the y axis
    :param inf_lim: inferior limit for both axis
    :param sup_lim: superior limit for both axis
    :param sTitle: string containing the title
    :param vLabels: vector containing the point labels
    :param sXTitle: string containing the title for the x axis
    :param sYTitle: string containing the title for the y axis
    :param sCaption: string containing the title. Defaults to nothing
    :param sFigName: desired file name of the saved figure (without the extension).
        The figures are saved in the "Figuras" subdirectory.
    :param saveFig: boolean; whether to save graph or not in the 'Figuras' subdirectory
    :param nTextLimit: minimal distance to origin that a point has to have in order for the sector's name to be plotted
    :return:
        fig: matplotlib object
        Also, saves the plot (in pdf) to the "Figuras" subdirectory.
    """
    if not saveFig:
        return 0
    else:
        # Creating fig object
        fig, ax = plt.subplots(figsize=(10, 6))

        # Painting the outside of the plot with the color light blue
        fig.set_facecolor("#e6f2ff")

        # Scatter plot
        plt.scatter(x, y, c="black", zorder=3)
        plt.grid(zorder=0)
        # Setting Limits
        ax.set_xlim(inf_lim, sup_lim)
        ax.set_ylim(inf_lim, sup_lim)
        # Creating dashed h and vlines at 1
        plt.hlines(1, xmin=inf_lim, xmax=sup_lim, colors='black', linestyles='dashed')
        plt.vlines(1, ymin=inf_lim, ymax=sup_lim, colors='black', linestyles='dashed')
        # Point Labels
        for i, txt in enumerate(vLabels):
            if abs(1 - x[i]) > nTextLimit:
                ax.annotate(txt, (x[i] + 0.012, y[i] - 0.012))

        # Titles and Captions
        ax.set_title(sTitle, fontsize=18)
        ax.set_xlabel(sXTitle, fontsize=13)
        ax.set_ylabel(sYTitle, fontsize=13)

        # Creating caption
        plt.figtext(0.125, 0.005, sCaption,
                    wrap=True, horizontalalignment='left', fontsize=12)

        # Annotating Quadrants
        plt.figtext(0.13, 0.85, "Forte encadeamento para trás",
                    wrap=True, horizontalalignment='left', fontsize=11)
        plt.figtext(0.13, 0.12, "Fraco encadeamento",
                    wrap=True, horizontalalignment='left', fontsize=11)
        plt.figtext(0.805, 0.85, "Setor-Chave",
                    wrap=True, horizontalalignment='left', fontsize=11)
        plt.figtext(0.65, 0.12, "Forte encadeamento para frente",
                    wrap=True, horizontalalignment='left', fontsize=11)

        # Saving graph
        sFileName = "Figuras/" + sFigName + ".pdf"
        fig.savefig(sFileName, dpi=1200)
        # fig.show()

        return fig

### ============================================================================================

def leontief_open(mIC, mProduction, nSectors):
    """
    Calculates Leontief and the Direct Technical Coefficients matrices.
    :param mIC: array containing the intermediate consumption matrix
    :param mProduction: array containing the production vector
    :param nSectors: number of sectors in matrix
    :return:
        mA: technical coefficients matrix
        mLeontief: Leontief's matrix
    """
    ## Diagonal Matrix of total production
    mX_diag = np.diagflat(1 / mProduction)

    ## A matrix
    mA = mIC.dot(mX_diag)

    ## Calculating Leontief's Matrix
    # Identity
    mI = np.eye(nSectors)
    # Inverse
    mLeontief = np.linalg.inv(mI - mA)

    return mA, mLeontief

def leontief_closed(mTechnical, cConsumption, cRem, nSectors):
    """
    Calculates Leontief and the Direct Technical Coefficients matrices in the CLOSED model.
    :param mTechnical: technical coefficients matrix of the open model
    :param cConsumption: vector containing the consumption coefficients
    :param cRem: vector containing the remunerations coefficients
    :param nSectors: number of sector
    :return:
        mA: technical coefficients matrix
        mLeontief: Leontief's matrix
    """

    ## Concatenating A and coefficients vector
    # Horizontally (family consumption)
    mA = np.concatenate((mTechnical, cConsumption), axis=1)
    # Vertically (income generation)
    # Adding a 0 to the end of cConsumption in order to join the matrices
    cRem = np.append(cRem, [0])
    mA = np.vstack((mA, cRem))

    ## Calculating Leontief's Matrix
    # Identity
    mI = np.eye(nSectors + 1)
    # Inverse
    mLeontief = np.linalg.inv(mI - mA)

    return mA, mLeontief

def ghosh_supply(mIC, mProduction, nSectors):
    """
    Calculates Ghosh's and the Direct Technical Coefficients matrices (supply-side model)
    :param mIC: array containing the intermediate consumption matrix
    :param mProduction: array containing the production vector
    :param nSectors: number of sectors in matrix
    :return:
        mA: technical coefficients matrix
        mGhosh: Ghosh's matrix
    """
    ## Diagonal Matrix of total production
    mX_diag = np.diagflat(1 / mProduction)

    ## A matrix
    mA = mX_diag.dot(mIC)

    ## Calculating Ghosh's Matrix
    # Identity
    mI = np.eye(nSectors)
    # Inverse
    mGhosh = np.linalg.inv(mI - mA)

    return mA, mGhosh

def calc_multipliers(mDesired, mProduction, mLeontief_open, mLeontief_closed, vSectors, nSectors):
    """
    Calculates the model's multipliers
    :param mDesired: matrix to be used to calculate multipliers (labor or income)
    :param mProduction: matrix (nSectors x 1 array) containing production by sector
    :param mLeontief_open: Leontief Matrix (open model)
    :param mLeontief_closed: Leontief Matrix (closed model)
    :param vSectors: array containing sector's names
    :param nSectors: number of sectors
    :return:
        mMultipliers: matrix containing the multiplier by sector
    """
    ## Coeficient: how much of income/labor is necessary in order to produce 1 unit in each sector
    h = mDesired/mProduction
    # Reshaping (from 1D matrix to vector)
    h = np.reshape(h, -1)
    # Diagonal matrix
    h_diag = np.diagflat(h)
    # Employment/Income generator matrix (open model): given an increase of 1 in final demand of that sector,
    # how much labor/income is generated in the economy?
    G = h_diag.dot(mLeontief_open)
    # Employment generator matrix (closed model): included inducing effects of consumption expansion
    G_closed = h_diag.dot(mLeontief_closed[:nSectors, :nSectors])

    ## Simple multipliers
    M = np.sum(G, axis=0)
    ## Type I multipliers
    MI = M / h

    ## Total (truncated) labor multipliers
    MT = np.sum(G_closed, axis=0)
    ## Type II labor multipliers
    MII = MT / h

    ## Joining all multipliers in a single table
    mMultipliers = np.vstack((vSectors, M, MI, MT, MII)).T

    return mMultipliers

def calc_ipl(mDemand, mA, vSectors, nSectors):
    """
    Calculates the Pure HR Indices
    :param mDemand: array containing the final demand for each sector
    :param mA: technical coefficients matrix
    :param nSectors: number of sectors
    :param vSectors: vector containg sector's names
    :return:
        IPL: containing the mPL
    """

    ## Creating list in order to help with slicing
    truth = [True] * nSectors

    ## Creating array to store the components
    IPL = np.zeros([nSectors, 3], dtype=float)

    ## Reshaping demand array (from matrix to vector)
    mDemand = np.reshape(mDemand, -1)

    ## creating GHS indices; for info, see Vale, Perobelli, 2020, p. 91-93
    # As there is a matrix composition (see eq.39), we use three loops
    # With the three ifs, we can be sure to calculate Ajj correctly
    for s in range(nSectors):
        for i in range(nSectors):
            for j in range(nSectors):
                if s == i:
                    if i == j:  # Ajj area
                        # Boolean array to determine which sectors are the "rest of the economy"
                        # (sector i = j = s -> False)
                        other_sectors = np.array(truth)
                        other_sectors[i] = False
                        # Demand of sector j and of the rest of the economy
                        yj = mDemand[i]
                        yr = mDemand[other_sectors]

                        # Direct technical coefficient of sector i = j with respect to itself
                        Ajj = mA[i, j]
                        # Indirect and direct technical coefficient of sector i = j with respect to itself
                        DJ = 1/(1 - Ajj)

                        # Direct technical coefficients of sales made by sector i=j=s to the rest of the economy
                        Ajr = mA[i, other_sectors]
                        # Direct coefficients of inputs bought by sector i=j=s from the rest of the economy
                        Arj = mA[other_sectors, j]
                        # Direct technical coefficients of the rest of the economy
                        Arr = mA[np.ix_(other_sectors, other_sectors)]

                        # Leontief matrix considering only the rest of the economy
                        mI = np.eye(nSectors - 1)
                        DR = np.linalg.inv(mI - Arr)

                        # PBL Indicator: impact of production of sector j upon the rest of the economy
                        # excluding self-demand for inputs and the return of other economic sectors to sector i=j=s
                        PBL = np.sum(
                            DR.dot(Arj).dot(DJ).dot(yj)
                        )
                        # PFL indicator: impact of the production of the rest of the economy upon sector i=j=s
                        PFL = np.dot(DJ, Ajr).dot(DR).dot(yr)
                        PTL = PBL + PFL

                        IPuros = np.array([PBL, PFL, PTL])

        ## adding calculated indices to the aggregate table
        IPL[s, :] = IPuros

    ## Normalizing indexes
    # dividing each sector's index by the mean index of the economy to see which are the most important
    # If > 1, the sector is key/chained to the rest of the economy,
    # even when taking in consideration its level of production
    IPLmean = np.sum(IPL, axis=0) / nSectors
    IPLmean = np.diagflat(1 / IPLmean)
    # Dividing all indicators by that indicator's mean
    IPLNorm = IPL.dot(IPLmean)

    ## Adding sectors names
    # Changing sector names vector to 1D matrix
    vSectors = np.reshape(vSectors, (nSectors, 1))
    IPL = np.concatenate((vSectors, IPL), axis=1)
    IPLNorm = np.concatenate((vSectors, IPLNorm), axis=1)

    return IPL, IPLNorm

def influence_matrix(mA, increment, vSectors, nSectors, sFigName, saveFig=False):
    """
    Calculates the influence matrix, detailed by sector (see Vale, Perobelli, p. 98-103)
    :param mA: technical coefficient matrix
    :param increment: integer containing the increment
    :param vSectors: vector containing name of sectors
    :param nSectors: number of sectors
    :param saveFig: boolean; whether to save heatmaps or not in the 'Figuras' subdirectory
    :param sFigName: desired file name of the saved figure (without the extension).
        The figures are saved in the "Figuras" subdirectory.
    :return:
        mInfluence: influence matrix
    """

    ## Calculating Leontief's Matrix
    mI = np.eye(nSectors)
    mB = np.linalg.inv(mI - mA)

    # Creating empty matrix for the increments and influences
    mIncrement = np.zeros([nSectors, nSectors], dtype=float)
    mInfluence = np.zeros([nSectors, nSectors], dtype=float)

    # Influence matrix
    for r in range(nSectors):
        for c in range(nSectors):
            ## Adding increment to the element that represent's sales of sector r
            # that will be used in the production of sector c
            mIncrement[r, c] = increment
            mA_increment = mA + mIncrement
            # Leontief's matrix with the increment in only that sector's relationshi[
            mB_increment = np.linalg.inv(mI - mA_increment)

            # Calculating influence area
            mFE = (mB_increment - mB) / increment
            S = np.sum(mFE * mFE)

            # Filling the influence matrix
            mInfluence[r, c] = S
            # Resetting the increment
            mIncrement[r, c] = 0

    ## Checking whether to draw graphs or not
    # The darker the color, the larger the importance of the link
    # between sectors i (selling) and j (buying input for production)
    # Therefore, it the sector's row is darker, the larger impact it has selling goods
    # if the sector's column is darker, the larger impact it has buying goods
    if not saveFig:
        return mInfluence, 0, 0
    else:
        print(f"Starting influence matrix graphs... ({datetime.datetime.now()})")
        ## Getting mean and standard deviation
        mean_Inf = np.mean(mInfluence)
        std_Inf = np.std(mInfluence, ddof=1)

        ## Creating new column to facilitate visualization
        situation_Inf = np.zeros([nSectors, nSectors], dtype=float)
        for r in range(nSectors):
            for c in range(nSectors):
                situation_Inf[r, c] = np.where(
                    mInfluence[r, c] < mean_Inf,
                    0,
                    np.where(
                        mInfluence[r, c] < mean_Inf + std_Inf,
                        1,
                        np.where(
                            mInfluence[r, c] < mean_Inf + 2*std_Inf,
                            2,
                            3
                        )
                    )
                )

        labels = ["< Média", "< Média + DP", "< Média + 2DP", "> Média + 2DP"]

        ### Creating a heatmap plot - continuous values
        # Creating fig object
        fig, ax = plt.subplots(figsize=(10, 10))

        # Creating heat map
        ax.imshow(mInfluence, cmap="Greys")

        # Painting the outside of the plot with the color light blue
        fig.set_facecolor("#e6f2ff")

        # Setting title
        ax.set_title("Campo de Influência - 2015", fontsize=18)
        ax.set_xlabel("Setores", fontsize=13)
        ax.set_ylabel("Setores", fontsize=13)

        # Adjusting axis
        plt.tick_params(axis='x', which='major', labelsize=12)
        plt.tick_params(axis='y', which='major', labelsize=12)
        ax.set_xticks(np.arange(nSectors))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_yticks(np.arange(nSectors))
        # ... and label them with the respective list entries
        ax.set_xticklabels(vSectors)
        ax.set_yticklabels(vSectors)

        ## Saving the figure
        sFileName = "Figuras/" + sFigName + ".pdf"
        fig.savefig(sFileName, dpi=1200)

        ### Creating a heatmap plot - continuous values
        # Creating fig_disc object
        fig_disc, ax = plt.subplots(figsize=(11, 10))

        # Creating heat map
        im = ax.imshow(situation_Inf, cmap="Greys")
        # Creating discrete colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(labelsize=12)

        # Painting the outside of the plot with the color light blue
        fig_disc.set_facecolor("#e6f2ff")

        # Setting title
        ax.set_title("Campo de Influência - 2015", fontsize=18)
        ax.set_xlabel("Setores", fontsize=13)
        ax.set_ylabel("Setores", fontsize=13)

        # Adjusting axis
        plt.tick_params(axis='x', which='major', labelsize=12)
        plt.tick_params(axis='y', which='major', labelsize=12)
        ax.set_xticks(np.arange(nSectors))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        ax.set_yticks(np.arange(nSectors))
        # ... and label them with the respective list entries
        ax.set_xticklabels(vSectors)
        ax.set_yticklabels(vSectors)

        ## Saving the figure
        sFileName = "Figuras/" + sFigName + "_Discreto.pdf"
        fig_disc.savefig(sFileName, dpi=1200)

        return mInfluence, fig, fig_disc

def extraction(mA, mA_supply, mProduction, mDemand, mSP, vSectors, nSectors):
    """
    Calculates extraction coefficients for each sector (backward and forward looking)
    :param mA: direct technical coefficient's matrix
    :param mA_supply: leontief's matrix (supply-side model)
    :param mProduction: production matrix
    :param mDemand: demand matrix
    :param mSP: payment sector matrix/vector
    :param vSectors: vector containing sector's names
    :param nSectors: number of sectors
    :return:
        mExtractions: table with the extraction indicators for each sector
    """

    ## Matrices structure
    BL_extrac = np.zeros([nSectors], dtype=float)
    FL_extrac = np.zeros([nSectors], dtype=float)

    # Identity
    mI = np.eye(nSectors)

    # For info, see Vale, Perobelli, p. 105-110
    for r in range(nSectors):
        for c in range(nSectors):
            ## Creating direct technical coefficients matrix WITHOUT sector c (buying - Backwards)
            # Creating a copy in order to not modify mA matrix
            mABL = np.copy(mA)
            mABL[:, c] = 0
            # Solving for Leontief's matrix without sector c buying
            mBBL = np.linalg.inv(mI - mABL)
            # Using the definition of Leontief's matrix to find the new production vector
            mXBL = mBBL.dot(mDemand)
            # Checking total production loss
            tbl = np.sum(mProduction) - np.sum(mXBL)
            BL_extrac[c] = tbl
            # Checking production loss with respect to the whole economy production (% terms)
            BL_extracp = BL_extrac / np.sum(mProduction) * 100

            ## Creating direct technical coefficients matrix WITHOUT sector r in the supply model (selling - Forwards)
            # Creating a copy in order to not modify mA_supply matrix
            mFFL = np.copy(mA_supply)
            mFFL[r, :] = 0
            # Solving for Ghosh's matrix without sector r selling
            mGFL = np.linalg.inv(mI - mFFL)
            # Using the definition of Ghosh's matrix to find the new production vector
            mXFL = mSP.T.dot(mGFL)
            # Checking total production loss
            tfl = np.sum(mProduction) - np.sum(mXFL)
            FL_extrac[r] = tfl
            # Checking production loss with respect to the whole economy production (% terms)
            FL_extracp = FL_extrac / np.sum(mProduction) * 100

    # Joining all vectors
    mExtractions = np.vstack((vSectors, BL_extrac, FL_extrac, BL_extracp, FL_extracp)).T

    return mExtractions

### ============================================================================================

def write_data_excel(FileName, vSheetName, vDataSheet):
    """
    Writes data to an Excel file (with multiple sheets) in the "Output" directory.
    :param FileName: name of the desired file
    :param vSheetName: array containing the desired names of the sheets
    :param vDataSheet: array containing the data to be written
    :return: Nothing; an Excel file is written in the "Output" directory.
    """

    Writer = pd.ExcelWriter('./Output/' + FileName, engine='openpyxl')
    df = []
    for each in range(len(vSheetName)):
        df.append(vDataSheet[each])
        df[each].to_excel(Writer, vSheetName[each], header=True, index=True)
    Writer.save()
