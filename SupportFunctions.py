"""
Auxiliary functions for Main.py (in order to keep the main code clean).
"""

## Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns


### ============================================================================================

def read_ibge(sFileMIP, sFileTRU):
    """
    Reads IBGE's Input Output File and the Added Value sheet from TRU's and returns the necessary matrices.
    :param sFileMIP: Excel file with IBGE's input-output tables
    :param sFileTRU: Excel file with IBGE's Added Value table
    :return:
        nSectors: integer containing the number of sectors
        vSectors: vector containing sector's names
        mZ: Intermediate Consumption matrix (nSectors x nSectors)
        mY: Demand vector (nSectors x 1)
        mX: Production vector (nSectors x 1)
        mC: Family Consumption vector (nSectors x 1)
        mV: Added Value vector (nSectors x 1)
        mR: Work Remunerations vector (nSectors x 1)
        mE: Employment vector (nSectors x 1)
        mSP: Payment Sector vector (nSectors x 1)
    """

    ## Reading table 03 (Intermediate Consumption and Final Demand - Base Prices)
    mip = pd.read_excel(sFileMIP, sheet_name="03")
    # Dropping "empty cells"
    mip = mip.dropna()
    # Creating an array
    m = mip.to_numpy()

    # Getting number of sectors
    nSectors = mip.shape[0]
    ## Getting sector's Names
    vSectors = m[:, 1]

    ## Getting Intermediate Consumption
    mZ = m[:, 3:nSectors + 3].astype(float)
    ## Getting Final Demand
    mY = m[:, -2]
    mY = np.reshape(mY, (nSectors, 1)).astype(float)
    ## Getting Production
    mX = m[:, -1]
    mX = np.reshape(mX, (nSectors, 1)).astype(float)
    ## Getting Family Consumption
    mC = m[:, -5]
    mC = np.reshape(mC, (nSectors, 1)).astype(float)

    ## Reading added value sheet in the TRU's Excel file
    tru = pd.read_excel(sFileTRU, sheet_name="VA")
    # Dropping "empty cells"
    tru = tru.dropna()
    # Creating an array
    t = tru.to_numpy()

    ## Getting Added Value
    mV = t[0, 1:nSectors + 1]
    mV = np.reshape(mV, (nSectors, 1)).astype(float)
    ## Getting Work Income (Remunerations)
    mR = t[1, 1:nSectors + 1]
    mR = np.reshape(mR, (nSectors, 1)).astype(float)
    ## Getting employed
    mE = t[-1, 1:nSectors + 1]
    mE = np.reshape(mE, (nSectors, 1)).astype(float)

    ## Getting Payment Sector (Total Production - National Production in Base Prices)
    mSP = t[-2, 1:nSectors + 1] - np.sum(mZ, axis=0)
    mSP = np.reshape(mSP, (nSectors, 1)).astype(float)

    return nSectors, vSectors, mZ, mY, mX, mC, mV, mR, mE, mSP


def read_estimated_mip(sFileMIP, sSheetName="MIP"):
    """
    Reads the output file of "EstimaMIPNacional" and returns the necessary matrices/vectors
    :param sFileMIP: file containing the estimated I-O Matrix
    :param sSheetName: name of the sheet to be read; defaults to "MIP"
    :return:
        dfMIP: original estimated MIP
        nSectors: Integer containing the number of sectors \n
        vSectors: Vector containing sector's names \n
        mZ: Intermediate Consumption matrix (nSectors x nSectors) \n
        mY: Demand vector (nSectors x 1) \n
        mX: Production vector (nSectors x 1)  \n
        mC: Family Consumption vector (nSectors x 1) \n
        mV: Added Value vector (nSectors x 1) \n
        mR: Work Remunerations vector (nSectors x 1) \n
        mE: Employment vector (nSectors x 1) \n
        mSP: Payment Sector vector (nSectors x 1) \n
        vAVNames: Vector containing the names of the added value components \n
        mAddedValue: Added value matrix (14 x nSectors) \n
        vFDNames: Vector containing the names of the final demand components \n
        mFinalDemand: Final demand matrix (nSectors x 6) \n
    """

    ## Reading Excel File
    dfMIP = pd.read_excel(sFileMIP, sheet_name=sSheetName, index_col=0)

    # Converting to an array
    mMIP = dfMIP.to_numpy()

    # Retrieving index
    vRowNames = np.array(dfMIP.index)

    ## Getting number of sectors and sectors names
    # Logic: find index of total Intermediate Consumption column
    try:
        nSectors = dfMIP.columns.get_loc("Total de Consumo Intermedi??rio")
    except KeyError:
        nSectors = dfMIP.columns.get_loc("Total do Consumo Intermedi??rio")

    # Sector names
    vSectors = vRowNames[:nSectors]

    ## Getting names of added value and demand components
    # Added Value Initial Line
    nAV, = np.where(vRowNames == "Valor adicionado bruto ( PIB )")
    nRelativePositionAV = nAV[0] - len(vRowNames)  # usually, -15; for nSectors = 51, -13

    # Added Value Names
    vAVNames = np.array(dfMIP.index)[nRelativePositionAV:-1]

    # Final Demand Names
    vFDNames = dfMIP.columns[nSectors + 1:-2]

    ## Intermediate Consumption for sectors i and j
    mZ = mMIP[:nSectors, :nSectors]
    mZ = mZ.astype(float)
    ## Final Demand of sector i
    mY = mMIP[:nSectors, -2]
    mY = np.reshape(mY, (nSectors, 1)).astype(float)
    ## Production of sector i
    mX = mMIP[:nSectors, -1]
    mX = np.reshape(mX, (nSectors, 1)).astype(float)
    ## Family Consumption of sector i
    mC = mMIP[:nSectors, -5]
    mC = np.reshape(mC, (nSectors, 1)).astype(float)

    ## Total Added Value of sector j
    mV = mMIP[nRelativePositionAV, :nSectors]
    mV = np.reshape(mV, (nSectors, 1)).astype(float)
    ## Work Income (Remunerations) of sector j
    mR = mMIP[nRelativePositionAV + 1, :nSectors]
    mR = np.reshape(mR, (nSectors, 1)).astype(float)
    ## Employment of sector j
    mE = mMIP[-2, :nSectors]
    mE = np.reshape(mE, (nSectors, 1)).astype(float)

    ## Payment Sector of sector j
    # (Imports + Taxes + Added Value = Total Production - Domestic Production at base prices)
    mSP = np.concatenate((mMIP[nSectors + 1:nRelativePositionAV - 1, :nSectors],
                          mMIP[nRelativePositionAV:nRelativePositionAV + 1, :nSectors]), axis=0).astype(float)

    ## Full Added Value Matrix
    mAddedValue = mMIP[nRelativePositionAV:-1, :nSectors].astype(float)

    ## Full Demand Matrix
    mFinalDemand = mMIP[:nSectors, nSectors + 1:-2].astype(float)

    return dfMIP, nSectors, vSectors, mZ, mY, mX, mC, mV, mR, mE, mSP, vAVNames, mAddedValue, vFDNames, mFinalDemand


def correct_order(df):
    """
    For base 12 sector aggregation, fixes the output order
    :param df: DataFrame containing the disaggregated data
    :return: dfOrdered: Ordered DataFrame
    """
    # Getting Housing and Food values
    lFilter = df.index.str.contains("Alojamento|Alimenta????o")
    dfHousingFood = df[lFilter]

    # Dropping them for the original dataframe
    df = df[~lFilter]

    # Appending the values to the end
    dfOrdered = df.append(dfHousingFood)

    return dfOrdered


def read_deflator(nYear, nSectors, EstimaMIP=True):
    """
    Reads the Excel file containing price indexes by sector for intermediate consumption, final demand and production
    2010 = 100
    :param nYear: Desired year to get the price index
    :param nSectors: Number of sectors
    :param EstimaMIP: boolean; whether to get indexes calculated using "EstimaMIP" matrices
        EstimaMIP=True is the only option when working with more than 20 sectors
    :return:
        mZ_index: price indexes for intermediate consumption (nSectors x nSectors matrix)
        mY_index: price indexes for final demand (nSectors vector)
        mX_index: price indexes for production (nSectors vector)
    """

    ## Getting file path
    if EstimaMIP or ((EstimaMIP is False) and (nSectors > 20)):
        if nSectors > 68:
            sFile = f"./Input/Deflatores/Deflatores_EstimaMIP_68+.xlsx"
        else:
            sFile = f"./Input/Deflatores/Deflatores_EstimaMIP_{nSectors}.xlsx"
    else:
        sFile = f"./Input/Deflatores/Deflatores_Tru_{nSectors}.xlsx"

    ## Reading Excel files
    mZ_index = pd.read_excel(sFile, sheet_name="Consum_Intermed").to_numpy()
    mY_index = pd.read_excel(sFile, sheet_name="Demanda_Final").to_numpy()
    mX_index = pd.read_excel(sFile, sheet_name="Producao").to_numpy()

    ## Getting relevant year's data
    # Intermediate Consumption
    mZ_index = mZ_index[(nYear - 2010) * nSectors:(nYear + 1 - 2010) * nSectors, 2:]
    mZ_index = mZ_index.astype(float)
    # Final Demand
    mY_index = mY_index[nYear - 2010, 1:].astype(float)
    # Production
    mX_index = mX_index[nYear - 2010, 1:].astype(float)

    return mZ_index, mY_index, mX_index


### ============================================================================================

def abbreviate_sectors_names(vSectors):
    """
    Abbreviates sector's names in order for them to fit in the graphs
    :param vSectors: vector containing sector's names
    :return:
        vSectorsGraph: vector with the abbreviated sectors
    """

    # Creating empty list
    vSectors_Graph = []
    # Chained ifs
    for name in vSectors:
        if name.startswith("Agr"):
            new_name = "Agro."
        elif name.startswith("Ind??strias e"):
            new_name = "Ind\nExtr."
        elif name.startswith("Ind??strias de"):
            new_name = "Ind\nTransf."
        elif name.startswith("Eletricidade e g??s,"):
            new_name = "SIUP"
        elif name.startswith("Eletricidade e g??s") or name.startswith("Energia el??trica, g??s"):
            new_name = "Eletr.\nG??s"
        elif name.startswith("??gua"):
            new_name = "??gua"
        elif name.startswith("Constru????o"):
            new_name = "Const."
        elif name.startswith("Com??rcio"):
            new_name = "Com."
        elif name.startswith("Transporte"):
            new_name = "Transp."
        elif name.startswith("Alojamento"):
            new_name = "Aloj.\nAlim."
        elif name.startswith("Informa????o"):
            new_name = "Info."
        elif name.startswith("Atividades financeiras"):
            new_name = "Fin."
        elif name.startswith("Atividades imobili??rias"):
            new_name = "Imob."
        elif name.startswith("Atividades cient??ficas"):
            new_name = "Ci??ncia"
        elif name.startswith("Atividades adm"):
            new_name = "Ativ.\nAdmin."
        elif name.startswith("Administra????o"):
            new_name = "Adm.\nP??b."
        elif name == "Educa????o":
            new_name = "Educ."
        elif name == "Educa????o B??sica":
            new_name = "Educ.\nB??sica"
        elif name.startswith("Sa??de"):
            new_name = "Sa??de"
        elif name.startswith("Artes"):
            new_name = "Artes"
        elif name.startswith("Atividades art??sticas"):
            new_name = "Artes"
        elif name == "Outras atividades industriais":
            new_name = "Outras\nInd??str."
        elif name.startswith("Outras atividades"):
            new_name = "Outros\nServ."
        elif name.startswith("Outros servi??os"):
            new_name = "Outros\nServ."
        elif name == "Servi??os dom??sticos":
            new_name = "Serv.\nDom."
        elif name == "Educa????o p??blica - Educa????o B??sica":
            new_name = "Educ.\nP??b. (EB)"
        elif name == "Educa????o p??blica  - Ensino m??dio e Outros":
            new_name = "Educ.\nP??b. (EM)"
        elif name == "Educa????o p??blica - Ensino Superior":
            new_name = "Educ.\nP??b. (ES)"
        elif name == "Educa????o privada -  Educa????o B??sica":
            new_name = "Educ.\nPriv. (EB)"
        elif name == "Educa????o privada  - Ensino m??dio e Outros":
            new_name = "Educ.\nPriv. (EM)"
        elif name == "Educa????o privada - Ensino Superior":
            new_name = "Educ.\nPriv. (ES)"
        elif name.startswith("Armazenamento"):
            new_name = "Armaz.\nCorreio"
        elif name.startswith("Organiza????es associativas"):
            new_name = "Org.\nassociativas"
        elif name == "Fabrica????o de alimentos e bebidas":
            new_name = "Fabr.\nAlim.\nBeb."
        elif name.startswith("T??xteis"):
            new_name = "T??xteis"
        elif name == "Fabrica????o de madeira, celulose e produtos de papel":
            new_name = "Papel\nMadeira"
        elif name == "Refino de petr??leo, coquerias e biocombust??veis":
            new_name = "Refino\nPetr."
        elif name.startswith("Qu??micos"):
            new_name = "Qu??m."
        elif name.startswith("Metalurgia"):
            new_name = "Meta-\nlurgia"
        elif name == "Equipamentos eletroeletr??nicos e mec??nicos":
            new_name = "Equip.\nEletr.\nMec."
        elif name == "Fabrica????o de autom??veis e pe??as":
            new_name = "Auto.\nPe??as"
        elif name == "Servi??os prestados ??s empresas":
            new_name = "Serv.\nEmpresas"
        else:
            new_name = name

        ## Appending new name to abbreviated sector's list
        vSectors_Graph.append(new_name)

    return np.array(vSectors_Graph)


def bar_plot(vData, vXLabels, sTitle, sXTitle, sFigName,
             nY_Adjust=0.001, BarColor="#595959", bMean=False, bAnnotate=True, nDirectory=None):
    """
    Creates a styled bar plot containing the data
    :param vData: vector (1D array) containing the data to be plotted
    :param vXLabels: vector containing the x-axis labels
    :param sTitle: string containing the title
    :param sXTitle: string containing the title for the x-axis
    :param nY_Adjust: float that adjusts the height of the annotations. Defaults to 0.001.
    :param sFigName: desired file name of the saved figure (without the extension).
        The figures are saved in the "Figuras" subdirectory.
    :param BarColor: color(s) (string or list of length nSectors) to fill the bars. Defaults to gray.
    :param bMean: boolean; plot a line of the economy's average? Defaults to False.
    :param bAnnotate: whether to annotate bars or not. Defaults to True.
    :param nDirectory: which directory to save the figures to;
        defaults to None, which saves according to the length of vXLabels.
        Very specific to the case of aggregation and structural decomposition
    :return: Nothing; saves the plot (in pdf) to the "Figuras" subdirectory in 'Output'.
    """

    ## Creating fig object
    # Determining size based on the number of sectors
    tupleFigSize = (8, 4) if len(vXLabels) <= 20 else (10, 5)

    # Creating fig object
    fig, ax = plt.subplots(figsize=tupleFigSize)

    ## Creating bars with the respective colours
    # If a list is supplied, check if it has the same length as number of sectors
    if (isinstance(BarColor, list) and len(vXLabels) == len(BarColor)) or (isinstance(BarColor, str)):
        plt.bar(x=vXLabels, height=vData, color=BarColor, zorder=4)
    else:
        print("Supplied color list doesn't have the same size as the number of sectors. Painting bars grey.")
        plt.bar(x=vXLabels, height=vData, color="#595959", zorder=4)

    # Adding average line
    if bMean:
        ax.axhline(vData.mean(), color="green", linewidth=4, linestyle="--")

    # Creating grid and painting the outside of the plot with the color light blue
    ax.grid(zorder=0)
    fig.set_facecolor("#e6f2ff")

    # Setting title and adjusting axis
    ax.set_title(sTitle, fontsize=12)
    ax.set_xlabel(sXTitle, fontsize=10)
    plt.tick_params(axis='x', which='major', labelsize=6)
    plt.tick_params(axis='y', which='major', labelsize=6)

    # Annotating the bars (if requested)
    if bAnnotate:
        for patch, label in zip(ax.patches, vData):
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2, height + nY_Adjust,
                np.around(label, 2), ha="center", va="bottom", fontsize=6
            )

    ## Saving the figure
    # File name and extension (pdf)
    if nDirectory is None:
        sFileName = f"Output/Figuras_{len(vXLabels)}/" + sFigName + ".pdf"
    else:
        sFileName = f"Output/Figuras_{nDirectory}/" + sFigName + ".pdf"

    # Saving figure and closing it to avoid memory issues (matplotlib allows only 20 open figures)
    fig.savefig(sFileName, dpi=600, bbox_inches="tight")
    plt.close(fig)


def named_scatter_plot(x, y, inf_lim, sup_lim, sTitle, vLabels, sXTitle, sYTitle, sFigName,
                       bPureLinkages, nTextLimit=0.045, PointColor="black"):
    """
    Creates a styled scatter plot containing the data and labels
    :param x: data for the x-axis
    :param y: data for the y-axis
    :param inf_lim: inferior limit for both axis
    :param sup_lim: superior limit for both axis
    :param sTitle: string containing the title
    :param vLabels: vector containing the point labels
    :param sXTitle: string containing the title for the x-axis
    :param sYTitle: string containing the title for the y -xis
    :param sFigName: desired file name of the saved figure. The figures are saved in the "Figuras" subdirectory.
    :param bPureLinkages: boolean; pure linkages? Required because the classification of key sectors is different.
    :param nTextLimit: minimal distance to (1, 1) that a point has to have in order for the sector's name to be plotted
    :param PointColor: color(s) (string or list of length nSectors) to fill the bars. Defaults to black.
    :return: Nothing; saves the plot (in pdf) to the "Figuras" subdirectory.
    """

    ## Creating fig object
    tupleFigSize = (8, 8)
    fig, ax = plt.subplots(figsize=tupleFigSize)

    # Painting the outside of the plot with the color light blue and creating grid lines
    fig.set_facecolor("#e6f2ff")
    plt.grid(zorder=0)

    ## Scatter plot with desired colors
    # If a list is supplied, check if it has the same length as number of sectors
    if (isinstance(PointColor, list) and len(vLabels) == len(PointColor)) or (isinstance(PointColor, str)):
        plt.scatter(x, y, c=PointColor, zorder=3)
    else:
        print("Supplied color list doesn't have the same size as the number of sectors. Painting points grey.")
        plt.scatter(x, y, c="black", zorder=3)

    # Setting Limits
    ax.set_xlim(inf_lim, sup_lim)
    ax.set_ylim(inf_lim, sup_lim)

    # Point Labels
    for i, txt in enumerate(vLabels):
        # if the point is too close to the origin, don't annotate
        if (abs(1 - x[i]) ** 2 + abs(1 - y[i]) ** 2) ** 0.5 > 1.41 * nTextLimit:
            if "\n" in txt:  # replace new lines with spaces
                txt = txt.replace("\n", " ")
            ax.annotate(txt, (x[i] + 0.02, y[i] - 0.012))

    # Titles and Captions
    ax.set_title(sTitle, fontsize=12)
    ax.set_xlabel(sXTitle, fontsize=10)
    ax.set_ylabel(sYTitle, fontsize=10)

    ## Creating lines and annotating quadrants
    # Pure Linkages
    if bPureLinkages:
        # Creating line that determines key sectors (PTLN = (PFLN + PBLN)/2 > 1, i.e., y > 2 - x)
        x_line = np.linspace(0, 2, 10)
        y_line = 2 - x_line
        ax.plot(x_line, y_line, color='black', linestyle='dashed')

    # Traditional Linkages
    else:
        # Creating dashed horizontal and vertical lines at 1
        plt.hlines(1, xmin=inf_lim, xmax=sup_lim, colors='black', linestyles='dashed')
        plt.vlines(1, ymin=inf_lim, ymax=sup_lim, colors='black', linestyles='dashed')

        # Annotating Quadrants
        plt.figtext(0.13, 0.85, "Forte Encadeamento para Tr??s", wrap=True, horizontalalignment='left', fontsize=11)
        plt.figtext(0.78, 0.85, "Setor-Chave", wrap=True, horizontalalignment='left', fontsize=11)
        plt.figtext(0.58, 0.1125, "Forte Encadeamento para Frente", wrap=True, horizontalalignment='left', fontsize=11)

    # Saving graph
    sFileName = f"Output/Figuras_{len(vLabels)}/{sFigName}.pdf"
    fig.savefig(sFileName, dpi=600, bbox_inches="tight")


def influence_matrix_graph(mInfluence, vSectors, nSectors, sTitle, sFigName):
    """
    Graphs the influence matrix: the darker the color, the larger the importance of the link
    between sectors i (selling) and j (buying input for production)
    Therefore, if the sector's row is darker, the larger impact it has selling goods
    if the sector's column is darker, the larger impact it has buying goods
    :param mInfluence: Influence Matrix calculated using the influence_matrix function
    :param vSectors: vector containing sector's names (preferably abbreviated)
    :param nSectors: number of sectors
    :param sTitle: title of the figure
    :param sFigName: name of the file
    :return: Nothing; saves influence matrix in the 'Output' directory
    """

    ## Replacing new lines with spaces
    vSectors = [name.replace("\n", " ") for name in vSectors]
    # Printing Start
    print(f"Starting influence matrix graphs... ({datetime.datetime.now()})")

    ## Getting mean and standard deviation (with one less degree of freedom)
    nMeanInfluence = np.mean(mInfluence)
    nSTDInfluence = np.std(mInfluence, ddof=1)

    ## Creating categorized matrix
    mCategoriesInfluence = np.where(mInfluence < nMeanInfluence, 1,
                                    np.where(mInfluence < nMeanInfluence + nSTDInfluence, 2,
                                             np.where(mInfluence < nMeanInfluence + 2 * nSTDInfluence, 3, 4)))

    labels = ["Menor que a M??dia", "Entre a M??dia\ne a M??dia + DP",
              "Entre a M??dia + DP\ne a M??dia + 2DP", "Maior que a\nM??dia + 2DP"]

    ### Creating a heatmap plot - continuous values
    ## Creating fig object
    # Determining size based on the number of sectors
    tupleFigSize = (6, 6) if nSectors < 67 else (10, 10)
    nRotation = 45 if nSectors < 20 else 90

    # Creating fig object
    fig, ax = plt.subplots(figsize=tupleFigSize)

    ## Creating heatmap
    sns.heatmap(mInfluence, cmap="Greys", annot=False, cbar_kws=dict(label="Encadeamento da Rela????o entre os Setores"))

    # Modifying color bar fontsize
    ax.figure.axes[-1].yaxis.label.set_size(8)

    # Painting the outside of the plot with the color light blue
    fig.set_facecolor("#e6f2ff")

    # Setting title
    ax.set_title(sTitle, fontsize=12)
    ax.set_xlabel("Setores", fontsize=10)

    # Adjusting axis
    plt.tick_params(axis='x', which='major', labelsize=6)
    plt.tick_params(axis='y', which='major', labelsize=6)

    # Adjusting ticks and labeling them with the respective list entries
    plt.xticks(np.arange(nSectors) + 0.5, vSectors, rotation=nRotation)
    plt.yticks(np.arange(nSectors) + 0.5, vSectors, rotation=0, va="center")

    ## Saving the figure
    sFileName = f"Output/Figuras_{nSectors}/" + sFigName + ".pdf"
    fig.savefig(sFileName, dpi=600, bbox_inches="tight")

    ### Creating a heatmap plot - continuous values
    # Creating fig_disc object
    fig_disc, ax = plt.subplots(figsize=tupleFigSize)

    ## Creating heatmap
    sns.heatmap(mCategoriesInfluence, cmap=["#FFFFFF", "#CDCDCD", "#737373", "#000000"], annot=False)

    # Creating discrete colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([1.375, 2.125, 2.876, 3.625])
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)

    # Painting the outside of the plot with the color light blue
    fig_disc.set_facecolor("#e6f2ff")

    # Setting title
    ax.set_title(sTitle, fontsize=12)
    ax.set_xlabel("Setores", fontsize=10)

    # Adjusting axis
    plt.tick_params(axis='x', which='major', labelsize=6)
    plt.tick_params(axis='y', which='major', labelsize=6)

    # Adjusting ticks and labeling them with the respective list entries
    plt.xticks(np.arange(nSectors) + 0.5, vSectors, rotation=nRotation)
    plt.yticks(np.arange(nSectors) + 0.5, vSectors, rotation=0, va="center")

    ## Saving the figure
    sFileName = f"Output/Figuras_{nSectors}/" + sFigName + "_Discreto.pdf"
    fig_disc.savefig(sFileName, dpi=600, bbox_inches="tight")


### ============================================================================================

def leontief_open(mIC, vProduction, nSectors):
    """
    Calculates Leontief and the Direct Technical Coefficients matrices.
    :param mIC: array containing the intermediate consumption matrix
    :param vProduction: array containing the production vector
    :param nSectors: number of sectors in matrix
    :return:
        mA: technical coefficients matrix
        mLeontief: Leontief's matrix
    """

    ## Diagonal Matrix of total production by sector
    mX_diag = np.diagflat(1 / vProduction)

    ## A matrix (dividing each intermediate consumption by the total production of sector j)
    mA = mIC.dot(mX_diag)

    ## Calculating Leontief's Matrix
    # Identity
    mI = np.eye(nSectors)
    # Inverse
    mLeontief = np.linalg.inv(mI - mA)

    return mA, mLeontief


def closed_model_guilhoto(mFinalDemand, mAddedValue, nSectors, nColISFLSFConsumption, nColFamilyConsumption,
                          nRowRemunerations, nRowRM, nRowEOB):
    """
    Adds up both types of consumption and, using mixed and capital income, fixes remunerations
    in order that the identity income = consumption is now True
    :param mFinalDemand: ((nSectors + 5) x 6) Final Demand Components Matrix (sectors and imports + taxes)
    :param mAddedValue: (14 x nSectors) Added Value Components Matrix
    :param nSectors: number of sectors
    :param nColISFLSFConsumption: column number for ISFLSF consumption in final demand matrix
    :param nColFamilyConsumption: column number for family consumption in final demand matrix
    :param nRowRemunerations: row number for remunerations in added value matrix
    :param nRowRM: row number for mixed incomes in added value matrix
    :param nRowEOB: row number for capital incomes in added value matrix
    :return:
        mC_Guilhoto: (nSectors x 1) matrix containing the sum of both consumptions
        mR_Guilhoto: (nSectors x 1) matrix containing remunerations, mixed income and residual capital income
        nTotalConsumption: integer containing sum of family and ISFLSF consumption (including taxes and imports)
    """

    # Adding up family and ISFLSF consumption
    mC_Guilhoto = np.sum(mFinalDemand[:, [nColISFLSFConsumption, nColFamilyConsumption]], axis=1, keepdims=True)
    nTotalConsumption = np.sum(mC_Guilhoto)

    ## Adding up remunerations
    # Remunerations and mixed incomes (RM, in portuguese)
    mR_Guilhoto = np.sum(mAddedValue[[nRowRemunerations, nRowRM], :], axis=0, keepdims=True).T
    # Capital income
    mEOB = mAddedValue[nRowEOB, :].reshape((nSectors, 1))
    mEOB_Guilhoto = mEOB.copy()
    nTotalEOB = np.sum(mEOB)

    ## Checking difference between income and consumption
    nTotalDifference = np.sum(mC_Guilhoto) - np.sum(mR_Guilhoto)

    # If income < consumption, part of EOB has to be used as family income
    if nTotalDifference > 0:
        mR_Guilhoto += mEOB / nTotalEOB * nTotalDifference
        mEOB_Guilhoto += -mEOB / nTotalEOB * nTotalDifference

    return mC_Guilhoto[:nSectors, :], mR_Guilhoto, mEOB_Guilhoto, nTotalConsumption


def leontief_closed(mTechnical, cConsumption, cRem, nSectors):
    """
    Calculates Leontief and the Direct Technical Coefficients matrices in the CLOSED model.
    :param mTechnical: technical coefficients matrix of the open model
    :param cConsumption: vector containing the consumption coefficients
    :param cRem: vector containing the remunerations coefficients
    :param nSectors: number of sectors;
    :return:
        mA: technical coefficients matrix
        mLeontief: Leontief's matrix
    """

    ## Concatenating A and coefficients vector
    # Horizontally (family consumption)
    mA = np.concatenate((mTechnical, cConsumption), axis=1)
    # Vertically (income generation) (adding a 0 to the end of cConsumption in order to join the matrices)
    cRem = np.append(cRem, [0])
    mA_closed = np.vstack((mA, cRem))

    ## Calculating Leontief's Matrix
    # Identity
    mI = np.eye(nSectors + 1)
    # Inverse
    mLeontief_closed = np.linalg.inv(mI - mA_closed)

    return mA_closed, mLeontief_closed


def ghosh_supply(mIC, vProduction, nSectors):
    """
    Calculates Ghosh's and the Direct Technical Coefficients Matrices (supply-side model)
    :param mIC: array containing the intermediate consumption matrix
    :param vProduction: array containing the production vector
    :param nSectors: number of sectors in matrix
    :return:
        mF: technical coefficients matrix
        mGhosh: Ghosh's matrix
    """

    ## Diagonal Matrix of total production
    mX_diag = np.diagflat(1 / vProduction)

    ## F matrix
    # Instead of dividing by the production of sector j, we divide by production of sector i
    mF = mX_diag.dot(mIC)

    ## Calculating Ghosh's Matrix
    # Identity
    mI = np.eye(nSectors)

    # Inverse
    mGhosh = np.linalg.inv(mI - mF)

    return mF, mGhosh


def calc_multipliers(vInput, vProduction, mDirectCoef, mLeontief_open, mLeontief_closed, vSectors, nSectors):
    """
    Calculates the model's multipliers/generators for labor and income
    :param vInput: vector (nSectors x 1 array) to be used to calculate the multipliers (labor or income)
    :param vProduction: vector (nSectors x 1 array) containing production by sector
    :param mDirectCoef: Technical Coefficients Matrix (mA)
    :param mLeontief_open: Leontief Matrix (open model)
    :param mLeontief_closed: Leontief Matrix (closed model)
    :param vSectors: array containing sector's names
    :param nSectors: number of sectors
    :return: mMultipliers: matrix containing the multiplier by sector
    """

    ## Coefficients: how much of income/labor/taxes is necessary in order to produce 1 unit in each sector
    h = vInput / vProduction

    # Reshaping (from 1D matrix to vector)
    h = np.reshape(h, -1)

    # Diagonal matrix ("h_hat")
    h_diag = np.diagflat(h)

    # Employment/Income generator matrix (open model): given an increase of 1 in final demand of that sector,
    # how much labor/income/taxes is generated in the economy?
    G = h_diag.dot(mLeontief_open)

    # Employment/Income generator matrix (closed model): includes inducing effects of consumption and income expansion
    G_closed = h_diag.dot(mLeontief_closed[:nSectors, :nSectors])

    ## Simple multipliers
    M = np.sum(G, axis=0)

    ## Type I multipliers
    MI = np.zeros(nSectors)
    for nCoefficient in range(h.size):
        MI[nCoefficient] = M[nCoefficient] / h[nCoefficient] if h[nCoefficient] != 0 else 0

    ## Total (truncated) multipliers
    MT = np.sum(G_closed, axis=0)

    ## Type II multipliers
    MII = np.zeros(nSectors)
    for nCoefficient in range(h.size):
        MII[nCoefficient] = MT[nCoefficient] / h[nCoefficient] if h[nCoefficient] != 0 else 0

    ## Separating multipliers into direct, indirect and induced effects (for simple and total multipliers)
    DirectEffects = np.sum(h_diag.dot(mDirectCoef), axis=0)
    IndirectEffects = M - DirectEffects
    InducedEffects = MT - M

    ## Joining coefficients, multipliers and effects into a single table
    mMultipliers = np.vstack((vSectors, h, M, MI, MT, MII, DirectEffects, IndirectEffects, InducedEffects)).T

    return mMultipliers


def linkages(mLeontief, mGhosh, vColNames, nSectors, vSectors):
    """
    Calculates linkage indicators (backwards and forwards in Ghosh's and Leontief's models).
    :param mLeontief: Leontief matrix (see leontief_open);
    :param mGhosh: Ghosh's matrix (see ghosh_supply);
    :param vColNames: list containing column names;
    :param nSectors: integer containing number of sectors;
    :param vSectors: vector containing sector's names.
    :return: mIndLig: numpy array with
    """

    ## Meaning across lines (get column totals) in Leontief's matrix: backwards linkages
    # Indices show how much other sector's have to produce in order to meet a rise of 1 unit in sector's j final demand
    nBL_Bar = np.mean(mLeontief, axis=0)

    ## Adding across columns (get row/line totals) in Leontief's matrix: forward linkages
    # Indices show how much sector i has to produce in order to meet a rise of 1 in final demand of the whole economy
    nFL_Bar = np.mean(mLeontief, axis=1)

    ## Mean of all indirect and direct technical coefficients
    nMT = np.sum(mLeontief) / nSectors ** 2

    ## Backwards-Looking Index (Uj)
    BL = nBL_Bar / nMT
    ## Forward-Looking Index (Ui)
    FL = nFL_Bar / nMT

    ## Using Ghosh's matrix (supply-side model)
    MLG = np.mean(mGhosh, axis=1)
    G_star = np.sum(mGhosh) / nSectors ** 2
    FLG = MLG / G_star

    ## Joining all indices
    mIndLig = np.vstack((vSectors, BL, FL, FLG)).T

    ## Checking key sectors
    # Transforming into a dataframe
    dfIndLig = pd.DataFrame(mIndLig, columns=vColNames)
    dfIndLig['Setores-Chave'] = np.where(
        (dfIndLig['Para Tr??s'] >= 1) & (dfIndLig['Para Frente Ghosh'] >= 1),
        "Setor-Chave",
        "-"
    )

    return dfIndLig.to_numpy()


def calc_ipl(vDemand, mA, vSectors, nSectors):
    """
    Calculates the Pure HR Indices
    :param vDemand: array containing the final demand for each sector
    :param mA: technical coefficients matrix
    :param nSectors: number of sectors
    :param vSectors: vector containing sector's names
    :return:
        mIPL: matrix containing all the pure indexes (backwards, forwards and total)
        IPLNorm: matrix containing all the normalized indexes (mIPL divided by the indicator's mean)
    """

    ## Creating array to store the components
    mIPL = np.zeros([nSectors, 3], dtype=float)

    ## Reshaping demand array (from 1D matrix to vector)
    vDemand = np.reshape(vDemand, -1)

    ## Creating GHS indices; for info, see Vale, Perobelli, 2020, p. 91-93
    for s in range(nSectors):
        # Mask to help with slicing
        lRestEconomy = [True] * nSectors
        lRestEconomy[s] = False

        # Demand of sector j and of the rest of the economy
        yj = vDemand[s]
        yr = vDemand[lRestEconomy]

        # Direct technical coefficient of sector i == j with respect to itself
        Ajj = mA[s, s]
        # Indirect and direct technical coefficient of sector i = j with respect to itself (Leontief)
        DJ = 1 / (1 - Ajj)

        # Direct technical coefficients of inputs bought by sector i=j=s from the rest of the economy
        Ajr = mA[s, lRestEconomy]
        # Direct coefficients of inputs bought by the rest of the economy from sector i=j=s
        Arj = mA[lRestEconomy, s]
        # Direct technical coefficients of the rest of the economy
        # np.ix_: allows us to select a specified subset of rows and columns
        Arr = mA[np.ix_(lRestEconomy, lRestEconomy)]

        # Leontief matrix considering only the rest of the economy
        mI = np.eye(nSectors - 1)
        DR = np.linalg.inv(mI - Arr)

        # PBL Indicator: impact of production of sector j upon the production of the rest of the economy
        # excluding self-demand for inputs and the return of other economic sectors to sector i=j=s
        PBL = np.sum(np.dot(DR, Arj).dot(DJ).dot(yj))
        # PFL indicator: impact of the production of the rest of the economy upon sector i=j=s
        PFL = np.sum(np.dot(DJ, Ajr).dot(DR).dot(yr))
        # Creating total index (PTL Indicator) to check which sectors are the most dynamic
        PTL = PBL + PFL

        # Putting all indicators in an array and adding to the table
        mIPL[s, :] = np.array([PBL, PFL, PTL])

    ## Normalizing indexes
    # dividing each sector's index by the mean index of the economy to see which are the most important
    # If > 1, the sector is key to the rest of the economy, even when taking in consideration its level of production
    vIPLmean = np.mean(mIPL, axis=0)
    # Dividing all indicators by that indicator's mean
    IPLNorm = mIPL / vIPLmean[None, :]  # equivalent to mIPL.dot(np.diagflat(1/vIPLmean), where diagflat produces a 3x3)

    ## Adding sectors names
    # Changing sector names vector to 1D matrix and concatenating with the mIPL and IPLNorm tables
    vSectors = np.reshape(vSectors, (nSectors, 1))
    mIPL = np.concatenate((vSectors, mIPL), axis=1)
    IPLNorm = np.concatenate((vSectors, IPLNorm), axis=1)

    return mIPL, IPLNorm


def influence_matrix(mA, nIncrement, nSectors):
    """
    Calculates the influence matrix, detailed by sector (see Vale, Perobelli, p. 98-103)
    :param mA: technical coefficient matrix
    :param nIncrement: integer containing the increment
    :param nSectors: number of sectors
    :return: mInfluence: influence matrix
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
            ## Adding increment to the element that represents sales of sector r
            # that will be used in the production of sector c
            mIncrement[r, c] = nIncrement
            mA_increment = mA + mIncrement
            # Leontief's matrix with the increment in only that sector's relationship
            mB_increment = np.linalg.inv(mI - mA_increment)

            # Calculating influence area
            mFE = (mB_increment - mB) / nIncrement
            S = np.sum(mFE * mFE)

            # Filling the influence matrix
            mInfluence[r, c] = S
            # Resetting the increment
            mIncrement[r, c] = 0

    return mInfluence


def extraction(mA, mA_supply, vProduction, vFinalDemand, mSP, vSectors, nSectors):
    """
    Calculates extraction coefficients for each sector (backward and forward-looking)
    :param mA: direct technical coefficient's matrix
    :param mA_supply: leontief's matrix (supply-side model)
    :param vProduction: production vector
    :param vFinalDemand: final demand vector
    :param mSP: payment sector matrix
    :param vSectors: vector containing sector's names
    :param nSectors: number of sectors
    :return: mExtractions: table with the extraction indicators for each sector
    """

    ## Matrices structure
    BL_extrac = np.zeros([nSectors], dtype=float)
    FL_extrac = np.zeros([nSectors], dtype=float)

    # Identity
    mI = np.eye(nSectors)

    # Adding up payment sector by sector
    vSP = np.sum(mSP, axis=0)

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
            mXBL = mBBL.dot(vFinalDemand)
            # Checking total production loss
            tbl = np.sum(vProduction) - np.sum(mXBL)
            BL_extrac[c] = tbl
            # Checking production loss with respect to the whole economy production (% terms)
            BL_extracp = BL_extrac / np.sum(vProduction) * 100

            ## Creating direct technical coefficients matrix WITHOUT sector r in the supply model (selling - Forwards)
            # Creating a copy in order to not modify mA_supply matrix
            mFFL = np.copy(mA_supply)
            mFFL[r, :] = 0
            # Solving for Ghosh's matrix without sector r selling
            mGFL = np.linalg.inv(mI - mFFL)
            # Using the definition of Ghosh's matrix to find the new production vector
            mXFL = vSP.T.dot(mGFL)
            # Checking total production loss
            tfl = np.sum(vProduction) - np.sum(mXFL)
            FL_extrac[r] = tfl
            # Checking production loss with respect to the whole economy production (% terms)
            FL_extracp = FL_extrac / np.sum(vProduction) * 100

    # Joining all vectors
    mExtractions = np.vstack((vSectors, BL_extrac, FL_extrac, BL_extracp, FL_extracp)).T

    return mExtractions


### ============================================================================================

def format_shock_sheet(vNew, vOld, vDelta, vPercentage, vSectors, vColNames):
    """
    Using the calculated shocks, returns a DataFrame with the formatted data
    :param vNew: new vector (after shock)
    :param vOld: original/mip vector
    :param vDelta: computed change between the above vectors
    :param vPercentage: computed percentage change between vNew and vOld
    :param vSectors: vector containing sector names
    :param vColNames: vector containing column names for the sheet
    :return: dfSheet: dataframe containing formatted data with impacts of the shock for each sector
    """

    ## Concatenating horizontally
    mSheet = np.vstack((vNew, vOld, vDelta, vPercentage)).T

    ## Calculating total row
    vTotal = np.sum(mSheet, axis=0)

    # Total percentage change
    vTotal[3] = vTotal[2] / vTotal[1]

    # Reshaping and concatenating
    totalV = np.reshape(vTotal, (1, 4))
    mSheet = np.concatenate((mSheet, totalV), axis=0)

    ## Formatting percentage column
    mSheet = mSheet.astype('object')
    mSheet[:, 3] = [f"{round(100 * i, 3)}%" for i in mSheet[:, 3]]
    mSheet[:, 3] = [f"{i.replace('.', ',')}" for i in mSheet[:, 3]]

    ## Index column
    vSectors_Sheet = np.append(vSectors, "Total")

    ## Creating DataFrame
    dfSheet = pd.DataFrame(mSheet, columns=vColNames, index=vSectors_Sheet)

    return dfSheet


def calculate_total_impact(vNew, vOld, vDelta):
    """
    For each matrix/vector, calculate the total impact for the whole economy
    :param vNew: vector/matrix containing the impact of the shock
    :param vOld: original vector/matrix
    :param vDelta: change between the two vector/matrices above
    :return: vTotal: vector containing totals before and after the shock, including absolute and percentage changes
    """

    ## Adding up each vector and concatenating vertically
    vTotal = np.hstack((np.sum(vNew), np.sum(vOld), np.sum(vDelta), 0))

    # Percentage column
    vTotal[3] = vTotal[2] / vTotal[1]

    return vTotal


def aggregate_shocks(dfData, nSectorsAggreg, sAggregFile, sAggregSheet, Aggreg, Add_Sectors,
                     lAdd_Sectors, lDisaggreg_Sectors, lAdd_Names, vSectors):
    """
    Having done all the calculations and formatting for the shock, aggregate/bundle up results
    :param dfData: DataFrame to be aggregated
    :param nSectorsAggreg: number of base sectors to aggregate to
    :param sAggregFile: name of the file containing aggregation correspondence
    :param sAggregSheet: name of the sheet within sAggregFile
    :param Aggreg: Aggregate?
    :param Add_Sectors: Bundle Up Sectors?
    :param lAdd_Sectors: Vector containing the indexes of the sectors to be added up
    :param lDisaggreg_Sectors: Vector containing the indexes of the sectors to NOT be aggregated
    :param lAdd_Names: Vector containing the names of the "new" sectors that are created when bundling
    :param vSectors: Vector containing sector names
    :return: dfData: Aggregated DataFrame
    """

    ## Checking to see if the number of base sectors is supported
    if nSectorsAggreg not in [4, 12, 20]:
        nSectorsAggreg = 20
        print("Invalid number for sector aggregation; only 4, 12 and 20 are supported. " +
              f"Aggregating for {nSectorsAggreg} sectors.")

    ## Reading aggregation file
    mAggreg = pd.read_excel(sAggregFile, sheet_name=sAggregSheet)
    # Dropping empty columns and converting to a numpy array
    mAggreg = mAggreg.dropna(axis=1)
    mAggreg = mAggreg.to_numpy()

    ## Selecting which column to iterate in
    # 4 sectors is column 2; 12 is column 3; 20 is column 4; all sectors is column 1
    if Aggreg:
        if nSectorsAggreg == 4:
            nColAggreg = 2
        elif nSectorsAggreg == 12:
            # An annoying thing will happen due to the way IBGE orders the sectors (particularly, housing and food);
            # if we want to keep them in order, we will have to reindex the dataframes used for the grouping-by later,
            # which is done by the correct_order function
            nColAggreg = 3
        else:
            nColAggreg = 4
    else:
        nColAggreg = 1

    ## Getting names of the sectors not to be aggregated and substituting them in the array
    if Aggreg:
        if lDisaggreg_Sectors is not None:
            for number in lDisaggreg_Sectors:
                mAggreg[number, nColAggreg] = vSectors[number]

    ## Getting names of the sectors to be bundled together and substituting them in the array
    if Add_Sectors:
        if lAdd_Sectors is not None:  # redundant, but used for the sake of parallelism and error handling
            for number in lAdd_Sectors:
                # Index of the number in the array
                nIndex = lAdd_Sectors.index(number)
                mAggreg[number, nColAggreg] = lAdd_Names[nIndex]

    ## Getting full correspondence of sectors
    lNewSectors = mAggreg[:, nColAggreg]
    lNewSectors = np.append(lNewSectors, ["Total"])

    # Adding the aggregated sectors to the end of the dataframe
    dfData["Novos_Setores"] = lNewSectors

    # Correcting order when aggregating to base 12 sectors
    if nSectorsAggreg == 12:
        dfData = correct_order(dfData)

    ## Adding up and grouping by
    dfData = dfData.groupby("Novos_Setores", sort=False).sum()

    ## Recalculating percentages and formatting
    dfData["Impacto Relativo (%)"] = dfData.values[:, 2] / dfData.values[:, 1]
    dfData["Impacto Relativo (%)"] = [f"{round(100 * i, 3)}%" for i in dfData["Impacto Relativo (%)"]]
    dfData["Impacto Relativo (%)"] = [f"{i.replace('.', ',')}" for i in dfData["Impacto Relativo (%)"]]

    return dfData


### ============================================================================================
def write_data_excel(sDirectory, sFileName, vSheetName, vDataSheet):
    """
    Writes data to an Excel file (with multiple sheets) in the "Output" directory.
    :param sDirectory: directory to save spreadsheet
    :param sFileName: name of the desired file
    :param vSheetName: array containing the desired names of the sheets
    :param vDataSheet: array containing the data to be written
    :return: Nothing; an Excel file is written in the "Output" directory.
    """

    ## Creating Writer object (allows writing of multiple sheets into a single file)
    Writer = pd.ExcelWriter(sDirectory + sFileName, engine='openpyxl')
    # Lists to store dataframe
    lDataFrames = []

    ## For each dataframe,
    for nSheet in range(len(vSheetName)):
        ## Append to list
        lDataFrames.append(vDataSheet[nSheet])

        ## Writing to Excel
        lDataFrames[nSheet].to_excel(Writer, vSheetName[nSheet], header=True, index=True, freeze_panes=(1, 1))

    ## Saving file
    Writer.save()
