### ============================================================================================
### Passagem do R-Script desenvolvido por VALE, PEROBELLI (20200 para o Python
### ============================================================================================

## Importing necessary packages
import numpy as np
import pandas as pd
import datetime
import SupportFunctions_V1 as Support

## Only run if it's the main file (don't run on import)
if __name__ == '__main__':
    ## Years to be analyzed
    # Main year
    nAno = 2015
    # Year to be compared in the structural decomposition
    nAno_StructuralDecomposition = 2010
    
    # Deflating factor (for structural decomposition)
    # This exists to remove the influence of price fluctuation over time;
    # Ideally, a vector of price inflation by sector calculated using the TRU's is to be used!
    nDeflator = 71.13  # 2015=100, 2010=71.13
    
    ## Whether to create and save figures (slows down the code from 3-5s to about 40-45s, mainly due to influence areas)
    saveFig = False
    saveFigIndicator = " (WITH figures)..." if saveFig else " (WITHOUT figures)..."

    ## Defining variable to be used for captions in figures (if necessary)
    sCaption = f"Fonte: elaboração própria com dados da MIP do IBGE ({nAno})."
    sCaption_StrucuturalDecomposition = \
        f"Fonte: elaboração própria com dados da MIP do IBGE ({nAno_StructuralDecomposition} e {nAno})."

    # Defining file paths and names
    sPath = "./Input/"
    sFileName = f"MIP{nAno}_12s.xlsx"
    sFileName_StructuralDecomposition = f"MIP{nAno_StructuralDecomposition}_12s.xlsx"

    sFile = sPath + sFileName
    sFile_StructuralDecomposition = sPath + sFileName_StructuralDecomposition
    
    ### ============================================================================================

    ## Print start
    sTimeBegin = datetime.datetime.now()
    print("======================= INPUT OUTPUT INDICATORS - VERSION 1 =======================")
    print(f"Starting{saveFigIndicator} ({sTimeBegin})")

    ## Reading Excel Sheets (see "MIP2015_12s", sheet "Leia-me" for more details)
    # Intermediate Consumption
    mZ = Support.read_matrices(sFile, sSheet="Z")
    # Demand
    mY = Support.read_matrices(sFile, sSheet="y")
    # Production
    mX = Support.read_matrices(sFile, sSheet="x")
    # Added Value
    mV = Support.read_matrices(sFile, sSheet="v")
    # Work Incomes (Remunerations)
    mR = Support.read_matrices(sFile, sSheet="r")
    # Employed
    mE = Support.read_matrices(sFile, sSheet="e")
    # Family Consumption
    mC = Support.read_matrices(sFile, sSheet="c")
    # Payment Sector (Total Production - National Production = Imports + Taxes + Added Value)
    mSP = Support.read_matrices(sFile, sSheet="sp")
    # Sectors
    mSet = Support.read_matrices(sFile, sSheet="set")

    ## Number of sectors
    nSectors = mSet.shape[0]
    vSectors = np.reshape(mSet, -1)

    ### ============================================================================================

    ### Open Model
    ## Technical Coefficients and Leontief's Matrice
    """
    mA (Technical Coefficients) tells us the monetary value of inputs from sector i
    that sector j directly needs to output/produce 1 unit,
    On the other hand, mB tells us, given a increase of 1 monetary value in the demand for products of sector j,
    how much should the production of each sector increase
    """
    mA, mB = Support.leontief_open(mZ, mX, nSectors)

    ### Closed Model
    """
    The open model captures only the direct and indirect impacts connected to intersectoral technical relations
    of buying and selling products, leaving out the effects induced by changes in income and consumption.
    In order to capture these phenomena, the model must be "closed" in relation to the families, 
    turning household consumption into an endogenous variable
    """
    ## Coefficients
    # Consumption: what families consume of each sector in respect to the total income
    hC = mC / np.sum(mR)
    # Remunerations (transposed): percentage of each sector's production that becomes income (work added value)
    hR = mR / mX
    hR = hR.T

    ## A and Leontief's matrix in the closed model
    """
    In this case, Leontief's matrix shows, given a one unit increase in demand for products of sector j,
    what are the direct, indirect and induced prerequisites for each sector's production.
    Therefore, the coefficients of mB_closed are larger than those of mB and their difference can be interpreted as the
    induced impacts of household consumption expansion upon the production of each sector.
    """
    mA_closed, mB_closed = Support.leontief_closed(mA, hC, hR, nSectors)

    ### Supply-side model
    mA_supply, mGhosh = Support.ghosh_supply(mZ, mX, nSectors)

    ### ============================================================================================
    ### Production Multipliers
    ### ============================================================================================

    ## Simple Production Multipliers (open model)
    """
    Given an increase of 1 in the demand of sector i, how much output is generated in the economy?
    """
    MP = np.sum(mB, axis=0)

    ## Total Production Multiples (closed model)
    """
    Given an increase of 1 in the demand of sector i, how much output is generated 
    in the economy, including the induced effects of household consumption expansion?
    It can be decomposed into a series of different effects (see Vale, Perobelli, p.43):
        - Induced effect (mB_closed - mB)
        - Direct effect (mA)
        - Indirect effect (mB - mA)
    """
    MPT = np.sum(mB_closed[:, :nSectors], axis=0)
    mInducedEffect = np.sum(mB_closed[:, :nSectors], axis=0) - np.sum(mB, axis=0)
    mDirectEffect = np.sum(mA, axis=0)
    mIndirectEffect = np.sum(mB, axis=0) - np.sum(mA, axis=0)
    mTotalEffectOpenModel = mDirectEffect + mIndirectEffect

    ## Total Truncated Production Multiples (closed model)
    """
    Given an increase of 1 in the demand of sector i, how much output is generated 
    in the economy, including the induced effects of household consumption expansion 
    (but only considerating the productive sectors; in other words, not considering
    the direct impact of household consumption on GDP, but only its induced effects)?
    """
    MPTT = np.sum(mB_closed[:nSectors, :nSectors], axis=0)
    mInducedEffectTrunc = np.sum(mB_closed[:nSectors, :nSectors], axis=0) - np.sum(mB, axis=0)

    ## Creating array with all multipliers
    mProdMultipliers_Col_Names = [
        "Setor", "Multiplicador Simples de Produção", "Multiplicador Total de Produção",
        "Multiplicador Total de Produção Truncado", "Efeito Induzido", "Efeito Direto", "Efeito Indireto"
    ]
    ## Creating table with all multipliers
    mProdMultipliers = np.vstack((vSectors, MP, MPT, MPTT, mInducedEffect, mDirectEffect, mIndirectEffect)).T

    ## Plotting all multipliers
    figSimpleProdMult = Support.bar_plot(
        vData=mProdMultipliers[:, 1], vXLabels=vSectors,
        sTitle="Multiplicadores Simples de Produção - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Simples_Producao", saveFig=saveFig
    )
    figTotalProdMult = Support.bar_plot(
        vData=mProdMultipliers[:, 2], vXLabels=vSectors,
        sTitle="Multiplicadores Totais de Produção - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Totais_Producao", saveFig=saveFig
    )
    figTotalTruncMult = Support.bar_plot(
        vData=mProdMultipliers[:, 3], vXLabels=vSectors,
        sTitle="Multiplicadores Totais de Produção Truncados - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Totais_Producao_Truncados", saveFig=saveFig
    )

    ### ============================================================================================
    ### Labor Multipliers
    """
    In line with the production multipliers, the simple labor multipliers tell us how many jobs 
    The total truncated labor multipliers, in turn, includes the induced effects of consumption expansion
    are generated (directly and indirectly) when there is a 1 million unit increase in demand for sector's j products
    Type I multipliers tell us how many jobs are directly and indirectly generated for each job directly created
    Type II multipliers tell us how many jobs are directly, indirectly and "inducedly" generated for each direct job
    """
    ### ============================================================================================

    ## Creating array with all multipliers
    mEmpMultipliers_Col_Names = [
        "Setor", "Multiplicador Simples de Emprego", "Multiplicador de Emprego Tipo I",
        "Multiplicador Total de Emprego (truncado)", "Multiplicador de Emprego Tipo II"
    ]
    ## Creating table with all multipliers
    mEmpMultipliers = Support.calc_multipliers(mE, mX, mB, mB_closed, vSectors, nSectors)

    ## Plotting all multipliers
    figSimpleEmpMult = Support.bar_plot(
        vData=mEmpMultipliers[:, 1], vXLabels=vSectors,
        sTitle="Multiplicadores Simples de Emprego - 2015", sXTitle="Setores",
        sCaption=f"{sCaption} Nota: por R$ milhão.",
        sFigName="Multiplicadores_Simples_Emprego", yadjust=0.1, saveFig=saveFig
    )
    figType1EmpMult = Support.bar_plot(
        vData=mEmpMultipliers[:, 2], vXLabels=vSectors,
        sTitle="Multiplicadores de Emprego (Tipo I) - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Tipo1_Emprego", yadjust=0.05, saveFig=saveFig
    )
    figTotEmpMult = Support.bar_plot(
        vData=mEmpMultipliers[:, 3], vXLabels=vSectors,
        sTitle="Multiplicadores Totais de Emprego (Truncados) - 2015", sXTitle="Setores",
        sCaption=f"{sCaption} Nota: por R$ milhão.",
        sFigName="Multiplicadores_Totais_Emprego", yadjust=0.1, saveFig=saveFig
    )
    figType2EmpMult = Support.bar_plot(
        vData=mEmpMultipliers[:, 4], vXLabels=vSectors,
        sTitle="Multiplicadores de Emprego (Tipo II) - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Tipo2_Emprego", yadjust=0.08, saveFig=saveFig
    )

    ### ============================================================================================
    ### Income Multipliers: Same interpretation as that of labor multipliers
    ### ============================================================================================

    ## Creating array with all multipliers
    mIncomeMultipliers_Col_Names = [
        "Setor", "Multiplicador Simples de Renda", "Multiplicador de Renda Tipo I",
        "Multiplicador Total de Renda (truncado)", "Multiplicador de Renda Tipo II"
    ]
    ## Creating table with all multipliers
    mIncomeMultipliers = Support.calc_multipliers(mR, mX, mB, mB_closed, vSectors, nSectors)

    ## Plotting all multipliers
    figSimpleIncomeMult = Support.bar_plot(
        vData=mIncomeMultipliers[:, 1], vXLabels=vSectors,
        sTitle="Multiplicadores Simples de Renda - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Simples_Renda", yadjust=0.0005, saveFig=saveFig
    )
    figType1IncomeMult = Support.bar_plot(
        vData=mIncomeMultipliers[:, 2], vXLabels=vSectors,
        sTitle="Multiplicadores de Renda (Tipo I) - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Tipo1_Renda", saveFig=saveFig
    )
    figTotIncomeMult = Support.bar_plot(
        vData=mIncomeMultipliers[:, 3], vXLabels=vSectors,
        sTitle="Multiplicadores Totais de Renda (Truncados) - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Totais_Renda", yadjust=0.0005, saveFig=saveFig
    )
    figType2IncomeMult = Support.bar_plot(
        vData=mIncomeMultipliers[:, 4], vXLabels=vSectors,
        sTitle="Multiplicadores de Renda (Tipo II) - 2015", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Multiplicadores_Tipo2_Renda", yadjust=0.002, saveFig=saveFig
    )
    
    ### ============================================================================================
    ### Índices de Ligação e Coeficientes de Variação
    """
    The indices show which sectors have larger chaining impacts in the economic, not only buying from other
    sectors in order to meet rises in its final demand ("backwards"/dispersion power - 
    how much the sector demands from others), but also producing to meet rising demand in the other 
    economic sectors ("forwards"/dispersion sensibility - how much the sector is demand by the others).
    We can normalize this impacts (dividing by the indicator's mean) in order to see which sectors
    are relatively more important/chained in the economy (norm. ind. > 1 -> bigger than the mean).
    """
    ### ============================================================================================

    ### Índices Hirschman-Rasmussen (HR)
    ## Meaning across lines (get column totals) in Leontief's matrix
    MC = np.mean(mB, axis=0)
    ## Adding across columns (get row/line totals) in Leontief's matrix
    ML = np.mean(mB, axis=1)
    ## Mean of all indirect and direct technical coefficients
    B_star = np.sum(mB) / nSectors**2

    ## Backwards-Looking Index (Uj)
    BL = MC/B_star
    ## Forward-Looking Index (Ui)
    FL = ML/B_star

    ## Using the Ghosh matrix (supply-side model)
    MLG = np.mean(mGhosh, axis=1)
    G_star = np.sum(mGhosh) / nSectors**2
    FLG = MLG / G_star

    ## Joining all indices
    mIndLig_Col_Names = ["Setor", "Para Trás", "Para Frente", "Para Frente Ghosh", "Setor-Chave"]
    mIndLig = np.vstack((vSectors, BL, FL, FLG)).T

    ## Checking key sectors
    # Transforming into a dataframe
    dfIndLig = pd.DataFrame(mIndLig, columns=mIndLig_Col_Names[:4])
    dfIndLig['Setores-Chave'] = np.where(
        (dfIndLig['Para Trás'] > 1) & (dfIndLig['Para Frente Ghosh'] > 1),
        "Setor-Chave",
        "-"
    )

    # Updating array
    mIndLig = dfIndLig.to_numpy()

    ## Creating a ScatterPlot
    figIndLig = Support.named_scatter_plot(
        x=mIndLig[:, 3], y=mIndLig[:, 1], inf_lim=0.5, sup_lim=1.5,
        sTitle="Índices de Ligação e Setores-Chave - 2015",
        sXTitle="Índice de Ligação para Frente  - Matriz de Ghosh", sYTitle="Índice de Ligação para Trás",
        vLabels=vSectors, sCaption=sCaption,
        sFigName="Indices_Ligacao_Setores_chave", saveFig=saveFig
    )

    ## Variance Coefficients
    # The lower the coefficent, the larger the number of sectors impacted by that sector's...
    # ... increase in final demand
    Vj = np.std(mB, axis=0, ddof=1) / MC
    # ... production/sales
    Vi = np.std(mB, axis=1, ddof=1) / ML

    # Joining into an aggregate table
    mVarCoef_Col_Names = ["Setor", "Vj", "Vi"]
    mVarCoef = np.vstack((vSectors, Vj, Vi)).T

    ### Índices Puros de Ligação (GHS)
    """
    As pointed out by Guilhoto (2009), the traditional indices don't take into consideration
    the production levels of each sector. The "pure" or "generalized" version doesn't have this problem.
    The backwards pure index (PBL) shows the impact of the production value of sector j upon the rest of the economy,
    excluding self-demand for its own inputs and the returns of the rest of the economy to the sector.
    The forward index (PFL) indicates the impact of the production value of the rest of the economy upon sector j 
    """

    ## Calculating IPL
    mIndPureLig_Col_Names = ["Setor", "PBL", "PFL", "PTL", "Setor-Chave"]
    mIndPureLig, mIndPureLigNorm = Support.calc_ipl(mY, mA, vSectors, nSectors)

    ## Checking key sectors
    # Transforming into a dataframe
    dfIndPureLigNorm = pd.DataFrame(mIndPureLigNorm, columns=mIndPureLig_Col_Names[:4])
    dfIndPureLigNorm['Setores-Chave'] = np.where(
        (dfIndPureLigNorm['PBL'] > 1) & (dfIndPureLigNorm['PFL'] > 1),
        "Setor-Chave",
        "-"
    )

    # Updating array
    mIndPureLigNorm = dfIndPureLigNorm.to_numpy()

    ## Creating a ScatterPlot
    figIndPureLig = Support.named_scatter_plot(
        x=mIndPureLigNorm[:, 2], y=mIndPureLigNorm[:, 1], inf_lim=0, sup_lim=4, nTextLimit=0.5,
        sTitle="Índices de Ligação Puros Normalizados e Setores-Chave - 2015",
        sXTitle="Índice Puro de Ligação para Frente Normalizados (PFLN)",
        sYTitle="Índice de Ligação para Trás Normalizados (PBLN)",
        vLabels=vSectors, sCaption=sCaption,
        sFigName="Indices_Ligacao_Puros_Setores_chave", saveFig=saveFig
    )

    ### ======================================================================================
    ### Campo de Influência
    """
    For more information, see Vale, Perobelli, 2020, p.98
    Although the indexes above show the importance of each sector in the economy as whole, it is difficult
    to visualize the main links through which this happens. Therefore, this concept shows how the changes
    in the direct technical coefficients are distributed throughout the economy, allowing us to see
    which relations among sectors are the most important within the productive sectors.
    In order to capture these individual changes, we use a singular increment matrix (with one element
    corresponding to the increment an the others, 0) an add that to the A matrix, calculating a new
    Leontief matrix. The influence of that sector's relation can be found subtracting the new Leontief
    by the default one and dividing by the increment
    """
    ### ============================================================================================

    ## Setting increment
    increment = 0.001

    ### Influence Matrix and Figure (see Vale, Perobelli, p. 98-103)
    mInfluence, figInfluenceCont, figInfluenceDisc = \
        Support.influence_matrix(mA, increment, vSectors, nSectors,
                                 saveFig=saveFig, sFigName="Campo_de_Influencia")

    ### ============================================================================================
    ### Hypothetical Extraction
    """
    What would happen if we extract a sector from economy? What if it doesn't buy anything from any other
    sectors or doesn't produce inputs from other sectors?
    This technique allows us to analyze the importance of the sector by eliminating it from the economy
    and measuring how much economic production decreases: the larger the interdependency of that sector
    within the economy, the larger the production loss.
    """
    ### ============================================================================================

    mExtractions_Col_Names = ["Setor", "BL", "FL", "BL%", "FL%"]
    mExtractions = Support.extraction(mA, mA_supply, mX, mY, mSP, vSectors, nSectors)

    ## Barplot with percentage indicators
    # BL % (production loss if the sector doesn't buy anything from the rest of economy,
    # relative to total economic production)
    figExtractionBackwards = Support.bar_plot(
        vData=mExtractions[:, 3], vXLabels=vSectors,
        sTitle="Perda de Produção por Extração Hipótetica - Estrutura de Compras (%)", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Extracao_Hipotetica_Compras", yadjust=0.01, saveFig=saveFig, sBarColor="green"
    )
    # FL % (production loss if the sector doesn't sell anything to the other economic sectors,
    # relative to total economic production)
    figExtractionForwards = Support.bar_plot(
        vData=mExtractions[:, 4], vXLabels=vSectors,
        sTitle="Perda de Produção por Extração Hipótetica - Estrutura de Vendas (%)", sXTitle="Setores",
        sCaption=sCaption,
        sFigName="Extracao_Hipotetica_Vendas", yadjust=0.01, saveFig=saveFig, sBarColor="green"
    )

    ### ============================================================================================
    ### Structural Decomposition - Open Model (p. 112)
    """
    ≈ Similar method to the oaxaca counterfactual decomposition
    The method allows the decomposition of the input-output relationship between two points in time
    into two effects: technical changes in sectors or changes in final demand
    This happens because of Leontief's matrix: between two given years (1 and 0), production changes can be written as
    mB(1)y(1) - mB(0)y(0), where y is the final demand.
    There are a series of possible decompositions:
        - Decomposition 1: ∆x = B(1)∆y + ∆By(0) = B(1)y(1) - B(1)y(0) + B(1)y(0) - B(0)y(0)
        B(1)y(1) - B(1)y(0): composition effect of change in demand
        B(1)y(0) - B(0)y(0): structural effect of changes in technological relations
        (changes in final demand pondered by technology of year 1 plus 
        technological changes pondered by demand of year 0)
        - Decomposition 2: ∆x = ∆By(1) + B(0)∆y = B(1)y(1) - B(0)y(1) + B(0)y(1) - B(0)y(0)
        B(0)y(1) - B(0)y(0): composition effect of change in demand
        B(1)y(1) - B(0)y(1): structural effect of backwards changes in technological relations
        (changes in final demand pondered by technology of year 0 plus 
        technological changes pondered by demand of year 1)
    
    Adding up the two decompositions, we can get the mean of the composition and structural change:
    ∆x = 0.5[B(1)∆y + ∆By(0)] + 0.5[∆By(1) + B(0)∆y] = 0.5∆B[y(0) + y(1)] + 0.5[B(0) + B(1)]∆y
        0.5∆B[y(0) + y(1)]: change due to technological modifications
        0.5[B(0) + B(1)]∆y: change due to demand shifts
    """
    ### ============================================================================================

    ## Reading 2015 (t=1) and 2010 (t=0) files
    # 2015 files are redundant, but for clarity will be read with the year as a suffix
    # Intermediate Consumption
    mZ1 = Support.read_matrices(sFile, sSheet="Z")
    mZ0 = Support.read_matrices(sFile_StructuralDecomposition, sSheet="Z")
    # Final Demand
    mY1 = Support.read_matrices(sFile, sSheet="y")
    mY0 = Support.read_matrices(sFile_StructuralDecomposition, sSheet="y")
    # Production
    mX1 = Support.read_matrices(sFile, sSheet="x")
    mX0 = Support.read_matrices(sFile_StructuralDecomposition, sSheet="x")
    
    ## Inflating 2010 prices to 2015's
    mZ0 = mZ0 / nDeflator * 100
    mY0 = mY0 / nDeflator * 100
    mX0 = mX0 / nDeflator * 100
    
    ## Direct technical coefficients and Leontief's matrix
    mA1, mB1 = Support.leontief_open(mZ1, mX1, nSectors)
    mA0, mB0 = Support.leontief_open(mZ0, mX0, nSectors)
    
    ## Changes in production
    deltaX = mX1 - mX0
    deltaX = np.reshape(deltaX, -1)
    ## Changes in Leontief's coefficients
    deltaB = mB1 - mB0
    totalB = mB1 + mB0
    ## Changes in demand
    deltaY = mY1 - mY0
    totalY = mY1 + mY0

    ## Decomposition of changes in production...
    # due to technological changes
    deltaTech = 0.5 * np.dot(deltaB, totalY)
    deltaTech = np.reshape(deltaTech, -1)
    # due to final demand shifts
    deltaDemand = 0.5 * np.dot(totalB, deltaY)
    deltaDemand = np.reshape(deltaDemand, -1)

    ## Joining all into one table
    mDecomposition_Col_Names = ["Setor", "Var. Produção", "Var. Tecnológica", "Var. Demanda Final"]
    mDecomposition = np.vstack((vSectors, deltaX, deltaTech, deltaDemand)).T

    # Getting Economy Total
    Total_Decomp = np.sum(mDecomposition, axis=0)
    Total_Decomp[0] = "Total"
    Total_Decomp = np.reshape(Total_Decomp, (1, 4))

    # Appending to end of the table
    mDecomposition = np.concatenate((mDecomposition, Total_Decomp), axis=0)
    mDecomposition_Index = np.append(vSectors, "Total")

    ## Bar-Plotting
    figDeltaX = Support.bar_plot(
        vData=mDecomposition[:nSectors, 1], vXLabels=vSectors,
        sTitle="Variação da Produção (R$ Milhões 2015)", sXTitle="Setores",
        sCaption=sCaption_StrucuturalDecomposition,
        sFigName="Variacao_Producao", saveFig=saveFig,
        sBarColor="darkblue", bAnnotate=False
    )
    figDeltaTech = Support.bar_plot(
        vData=mDecomposition[:nSectors, 2], vXLabels=vSectors,
        sTitle="Decomposição - Variação Tecnológica", sXTitle="Setores",
        sCaption=sCaption_StrucuturalDecomposition,
        sFigName="Variacao_Tecnologica", saveFig=saveFig,
        sBarColor="darkred", bAnnotate=False
    )
    figDeltaDemand = Support.bar_plot(
        vData=mDecomposition[:nSectors, 3], vXLabels=vSectors,
        sTitle="Decomposição - Variação da Demanda Final", sXTitle="Setores",
        sCaption=sCaption_StrucuturalDecomposition,
        sFigName="Variacao_DemandaFinal", saveFig=saveFig,
        sBarColor="dodgerblue", bAnnotate=False
    )

    ### ============================================================================================
    ### Exporting table to Excel
    ### ============================================================================================

    print("Writing Excel file...")

    ### Creating lots of dataframes
    vDataSheets = []
    vNameSheets = ["Mult_Prod"]

    # Production Multipliers
    vDataSheets.append(pd.DataFrame(mProdMultipliers[:, 1:], columns=mProdMultipliers_Col_Names[1:], index=vSectors))

    # Employment/Labor multipliers
    vNameSheets.append("Mult_Trab")
    vDataSheets.append(pd.DataFrame(mEmpMultipliers[:, 1:], columns=mEmpMultipliers_Col_Names[1:], index=vSectors))

    # Income multipliers
    vNameSheets.append("Mult_Renda")
    vDataSheets.append(
        pd.DataFrame(mIncomeMultipliers[:, 1:], columns=mIncomeMultipliers_Col_Names[1:], index=vSectors)
    )

    # "Índices de Ligação" (HR Indices)
    vNameSheets.append("Indices_Ligacao")
    vDataSheets.append(pd.DataFrame(mIndLig[:, 1:], columns=mIndLig_Col_Names[1:], index=vSectors))

    # Variance Coefficients
    vNameSheets.append("Coeficientes_Variacao")
    vDataSheets.append(pd.DataFrame(mVarCoef[:, 1:], columns=mVarCoef_Col_Names[1:], index=vSectors))

    # "Índices de Ligação Puros Normalizados" (GHS Indices)
    vNameSheets.append("Indices_Ligacao_Puros")
    vDataSheets.append(pd.DataFrame(mIndPureLigNorm[:, 1:], columns=mIndPureLig_Col_Names[1:], index=vSectors))

    # Influence Areas
    vNameSheets.append("Campo_de_Influencia")
    vDataSheets.append(pd.DataFrame(mInfluence, columns=vSectors, index=vSectors))

    # Hypothetical Extractions
    vNameSheets.append("Extracao_Hipotetica")
    vDataSheets.append(pd.DataFrame(mExtractions[:, 1:], columns=mExtractions_Col_Names[1:], index=vSectors))

    # Structural Decomposition
    vNameSheets.append("Decomposicao_Estrutural")
    vDataSheets.append(
        pd.DataFrame(mDecomposition[:, 1:], columns=mDecomposition_Col_Names[1:], index=mDecomposition_Index)
    )

    ## Writing Excel File to 'Output' directory
    Support.write_data_excel("Resultados_Python.xlsx", vNameSheets, vDataSheets)

    ### ============================================================================================

    ## Ending everything
    time_diff = datetime.datetime.now() - sTimeBegin
    print(f"All done! ({datetime.datetime.now()})")
    print(f"{time_diff.seconds} seconds passed.")
