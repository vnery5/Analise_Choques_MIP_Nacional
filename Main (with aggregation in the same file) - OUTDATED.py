### ============================================================================================
### Análise setorial a partir das matrizes estimadas pelo ESTIMAMIP
### O script também permite agregar a matriz/somar setores e calcular os indicadores com a nova estrutura setorial
### Autores: João Maria de Oliveira e Vinícius Nery (Ipea-DF; 2021)
### ============================================================================================

## Importing necessary packages
import numpy as np
import pandas as pd
import datetime
import math
import SupportFunctions as Support

## Only run if it's the main file (don't run on import)
if __name__ == '__main__':
    ## Matrix dimension
    # 1: 12 sectors
    # 2: 20 sectors
    # 3: 68 sectors
    # 9: more than 68 sectors
    # 0: other (specify number of sectors file below)
    nDimensao = 1

    ## Year to be analyzed
    nAno = 2015

    ## Whether to create and save figures
    saveFig = False  # True or False

    ### Aggregate sectors?
    Aggreg = False  # True or False
    sAggregFileName = "SetoresGrupo.xlsx"
    sAggregSheet = "SetorGrupo"

    ## If affirmative, how many base sectors would you like to aggregate to?
    nSectorsAggreg = 12  # 4, 12, 20

    ## Which sectors shall NOT be aggregated?
    # Use their index/number/position! (start: 0)
    # If you wish to get only the default 4/12/20, use None
    # lDisaggreg_Sectors = [61, 62, 63, 64, 65, 66]  # list of integers or None
    lDisaggreg_Sectors = None

    ### Add up/bundle sectors?
    Add_Sectors = False  # True or False

    ## Which sectors shall be bundled up together?
    # Use their index/number/position! (start: 0)
    lAdd_Sectors = [61, 62, 64, 65]  # list of integers
    # What shall them be named? Put the name for each index! (both lists have to be the same size)
    lAdd_Names = ["Educação Básica"] * 4

    ## Do a structural decomposition?
    doStructure = True  # True or False
    # Year to be compared in the structural decomposition
    nAno_StructuralDecomposition = 2010

    ### ============================================================================================

    ### Defining file paths and names
    if nDimensao == 9:
        nSectorsFile = "68+"
        invalidSectorNumber = ""
    elif nDimensao == 0:
        nSectorsFile = 4
        invalidSectorNumber = ""
    else:
        ## Dimensions -> sectors list
        listDimensions = [12, 20, 68]
        try:
            nSectorsFile = listDimensions[nDimensao - 1]
            invalidSectorNumber = ""
        except IndexError:
            nSectorsFile = listDimensions[2]
            invalidSectorNumber = "Couldn't find desired dimension. Running for 68 sectors."
            nDimensao = 3

    ## Esimated MIPs files
    sPathMIP = "./Input/MIPs Estimadas/Precos_Correntes/"
    sFileNameMIP = f"MIP_{nAno}_{nSectorsFile}.xlsx"
    sFileNameMIP_StructuralDecomposition = f"MIP_{nAno_StructuralDecomposition}_{nSectorsFile}.xlsx"

    # Joining path and file name
    sFileMIP = sPathMIP + sFileNameMIP
    sFileMIP_StructuralDecomposition = sPathMIP + sFileNameMIP_StructuralDecomposition

    # Sheet name
    sSheetNameMIP = "MIP"

    ## Aggregation file
    sPathInput = "./Input/"
    sAggregFile = sPathInput + sAggregFileName

    ## Defining indicators
    # Figures
    saveFigIndicator = " (WITH figures)..." if saveFig else " (WITHOUT figures)..."

    # Aggregation/adding
    if Aggreg and Add_Sectors:
        AggregIndicator = "_Agg_Add"
    elif Aggreg:
        AggregIndicator = "_Agg"
    elif Add_Sectors:
        AggregIndicator = "_Add"
    else:
        AggregIndicator = ""

    printAggregIndicator = f"Aggregations = {Aggreg}"
    printAddIndicator = f"Adding/Bundling = {Add_Sectors}"

    ## Defining variable to be used for captions in figures (if necessary)
    sCaption = f"Fonte: elaboração própria com dados do IBGE ({nAno})."
    sCaption_StrucuturalDecomposition = \
        f"Fonte: elaboração própria com dados do IBGE ({nAno_StructuralDecomposition} e {nAno})."
    ### ============================================================================================

    ### Print start
    sTimeBegin = datetime.datetime.now()
    print("======================= INPUT OUTPUT INDICATORS - VERSION 2 =======================")
    print(f"Starting for year = {nAno} and dimension = {nDimensao}{saveFigIndicator} ({sTimeBegin})")
    print(printAggregIndicator)
    print(printAddIndicator)
    print(invalidSectorNumber)

    ## Read necessary matrices and get number of sectors and sector's names
    nSectors, vSectors, mZ, mY, mX, mC, mV, mR, mE, mSP, vAVNames, mAddedValue, vFDNames, mFinalDemand = \
        Support.read_estimated_mip(sFileMIP,
                                   sSheetName=sSheetNameMIP,
                                   Aggreg=Aggreg, Add_Sectors=Add_Sectors,
                                   nSectorsAggreg=nSectorsAggreg, lDisaggreg_Sectors=lDisaggreg_Sectors,
                                   lAdd_Sectors=lAdd_Sectors, lAdd_Names=lAdd_Names,
                                   sAggregFile=sAggregFile, sAggregSheet=sAggregSheet
                                   )

    ### ============================================================================================

    ### Open Model
    ## Technical Coefficients and Leontief's Matrix
    """
    mA (Technical Coefficients) tells us the monetary value of inputs from sector i
    that sector j directly needs to output/produce 1 unit,
    On the other hand, mB tells us, given a increase of 1 monetary value in the demand for products of sector j,
    how much should the production of each sector increase
    """
    mA, mB = Support.leontief_open(mZ, mX, nSectors)

    ### Closed Model
    """
    The open model captures only the direct and indirect impacts connected to intersectorial technical relations
    of buying and selling products, leaving out the effects induced by changes in income and consumption.
    In order to capture these phenomena, the model must be "closed" in relation to the families, 
    turning household consumption into an endogenous variable alongside labor remunerations
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
    (but only considering the productive sectors; in other words, not considering
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

    ### ============================================================================================
    ### Labor Multipliers
    """
    In line with the production multipliers, the simple labor multipliers tell us how many jobs
    are generated (directly and indirectly) when there is a 1 million unit increase in demand for sector's j products
    The total truncated labor multipliers, in turn, includes the induced effects of consumption expansion
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

    ### Influence Matrix (see Vale, Perobelli, p. 98-103)
    mInfluence = Support.influence_matrix(mA, increment, nSectors)

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

    if doStructure:
        ## Reading t=1 and t=0 files
        # Disaggregation/aggregation is not yet supported due to the use of fixed price indexes :/
        # Year 1 (= as the rest of the analysis)
        nSectors1, vSectors1, mZ1, mY1, mX1, mC1, mV1, mR1, mE1, mSP1, \
            vAVNames1, mAddedValue1, vFDNames1, mFinalDemand1 = \
            Support.read_estimated_mip(sFileMIP)

        # Year 0
        nSectors0, vSectors0, mZ0, mY0, mX0, mC0, mV0, mR0, mE0, mSP0, \
            vAVNames0, mAddedValue0, vFDNames0, mFinalDemand0 = \
            Support.read_estimated_mip(sFileMIP_StructuralDecomposition)

        ### Inflating year 0 prices to year 1's
        ## Getting price indexes (2010 = 100)
        mZ_index1, mY_index1, mX_index1 = Support.read_deflator(nAno, nSectors1)
        mZ_index0, mY_index0, mX_index0 = Support.read_deflator(nAno_StructuralDecomposition, nSectors0)

        ## Deflating prices to 2010 levels
        # In order to maintain consistency, we must use the same indexes for every deflation
        # The vector of indexes of choice is that of production
        mZ1 = 100 * (mZ1 / mX_index1[:, None])  # using None in slicing allows us to divide the matrix by the vector
        mY1 = 100 * (mY1 / mX_index1[:, None])
        mX1 = 100 * (mX1 / mX_index1[:, None])

        mZ0 = 100 * (mZ0 / mX_index0[:, None])
        mY0 = 100 * (mY0 / mX_index0[:, None])
        mX0 = 100 * (mX0 / mX_index0[:, None])
        """
        # Deflating factor (for structural decomposition)
        # This exists to remove the influence of price fluctuation over time;
        # Ideally, a vector of price inflation by sector calculated using the TRU's is to be used!
        nDeflator = 71.13  # 2015=100, 2010=71.13
        
        ## Inflating 2010 prices to 2015's
        mZ0 = mZ0 / nDeflator * 100
        mY0 = mY0 / nDeflator * 100
        mX0 = mX0 / nDeflator * 100
        """
        """
        mZ1 = 100 * (mZ1 / mZ_index1)
        mY1 = 100 * (mY1 / mY_index1)
        mX1 = 100 * (mX1 / mX_index1)
    
        mZ0 = 100 * (mZ0 / mZ_index0)
        mY0 = 100 * (mY0 / mY_index0)
        mX0 = 100 * (mX0 / mX_index0)
        """
        ## Direct technical coefficients and Leontief's matrix
        mA1, mB1 = Support.leontief_open(mZ1, mX1, nSectors1)
        mA0, mB0 = Support.leontief_open(mZ0, mX0, nSectors0)

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
        mDecomposition_Col_Names = ["Setor", "Var. Produção", "Var. Tecnológica", "Var. Demanda Final",
                                    f"Índices de Preços da Produção Total {nAno}",
                                    f"Índices de Preços da Produção Total {nAno_StructuralDecomposition}"]
        mDecomposition = np.vstack((vSectors1, deltaX, deltaTech, deltaDemand, mX_index1, mX_index0)).T

        # Getting Economy Total
        Total_Decomp = np.sum(mDecomposition, axis=0)
        Total_Decomp[0] = "Total"
        Total_Decomp[[4, 5]] = "-"
        Total_Decomp = np.reshape(Total_Decomp, (1, 6))

        # Appending to end of the table
        mDecomposition = np.concatenate((mDecomposition, Total_Decomp), axis=0)
        mDecomposition_Index = np.append(vSectors1, "Total")

    ### ============================================================================================
    ### Exporting table to Excel
    ### ============================================================================================

    print(f"Writing Excel file... ({datetime.datetime.now()})")

    ### Creating lots of dataframes
    ## Lists that will contain the data and the names of the sheets
    vDataSheets = []
    vNameSheets = []

    ## Original Input-Output matrix
    vNameSheets.append("MIP_Original")
    dfMIP = pd.read_excel(sFileMIP, sheet_name=sSheetNameMIP, index_col=0)
    vDataSheets.append(dfMIP)

    ## Aggregated/bundled matrix (if aggregation was requested)
    if Aggreg or Add_Sectors:
        vNameSheets.append("MIP_Aggreg")

        # Determining shape of the matrix
        mMIP_Agg = np.zeros([nSectors + 22, nSectors + 9], dtype=float)
        # Adding intermediate consumption
        mMIP_Agg[:nSectors, :nSectors] = mZ
        # Payment Sector
        mMIP_Agg[nSectors + 1:nSectors + 6, :nSectors] = mSP[:-1, :]
        # Added Value
        mMIP_Agg[nSectors + 7:-1, :nSectors] = mAddedValue
        # Final Demand
        mMIP_Agg[:nSectors, nSectors + 1:-2] = mFinalDemand

        # Payment Sector of the final demand components
        nSectors_Orig = dfMIP.shape[0] - 22
        mMIP_Agg[nSectors + 1:nSectors + 6, nSectors + 1:-2] = \
            dfMIP.values[nSectors_Orig + 1:nSectors_Orig + 6, nSectors_Orig + 1:-2]

        # Total National Consumption Row
        mMIP_Agg[nSectors, :] = np.sum(mMIP_Agg[:nSectors, :], axis=0)
        # Total Consumption Row
        mMIP_Agg[nSectors + 6, :] = mMIP_Agg[nSectors, :] + np.sum(mMIP_Agg[nSectors + 1:nSectors + 6, :], axis=0)

        # Total National IC Column
        mMIP_Agg[:, nSectors] = np.sum(mMIP_Agg[:, :nSectors], axis=1)
        # Total Final Demand
        mMIP_Agg[:, -2] = np.sum(mMIP_Agg[:, nSectors + 1:-2], axis=1)
        # Total Demand/Production
        mMIP_Agg[:, -1] = mMIP_Agg[:, -2] + mMIP_Agg[:, nSectors]

        ## Converting to dataframe
        # Getting columns and index of the original MIP
        dfMIP_Col_Names = np.array(dfMIP.columns)
        dfMIP_Index = np.array(dfMIP.index)

        # Substituting the aggregated sectors
        dfMIP_Agg_Col_Names = np.append(vSectors, dfMIP_Col_Names[nSectors_Orig:])
        dfMIP_Agg_Index = np.append(vSectors, dfMIP_Index[nSectors_Orig:])

        # Creating DataFrame
        dfMIP_Agg = pd.DataFrame(mMIP_Agg, columns=dfMIP_Agg_Col_Names, index=dfMIP_Agg_Index)
        vDataSheets.append(dfMIP_Agg)

    # Production Multipliers
    vNameSheets.append("Mult_Prod")
    vDataSheets.append(pd.DataFrame(mProdMultipliers[:, 1:], columns=mProdMultipliers_Col_Names[1:], index=vSectors))

    # Employment/Labor multipliers
    vNameSheets.append("Mult_Trab")
    vDataSheets.append(pd.DataFrame(mEmpMultipliers[:, 1:], columns=mEmpMultipliers_Col_Names[1:], index=vSectors))

    # Income multipliers
    vNameSheets.append("Mult_Renda")
    vDataSheets.append(
        pd.DataFrame(mIncomeMultipliers[:, 1:], columns=mIncomeMultipliers_Col_Names[1:], index=vSectors)
    )

    # Variance Coefficients
    vNameSheets.append("Coef_Var")
    vDataSheets.append(pd.DataFrame(mVarCoef[:, 1:], columns=mVarCoef_Col_Names[1:], index=vSectors))

    # "Índices de Ligação" (HR Indices)
    vNameSheets.append("Ind_Lig")
    vDataSheets.append(pd.DataFrame(mIndLig[:, 1:], columns=mIndLig_Col_Names[1:], index=vSectors))

    # "Índices de Ligação Puros Normalizados" (GHS Indices)
    vNameSheets.append("Ind_Lig_Puros")
    vDataSheets.append(pd.DataFrame(mIndPureLigNorm[:, 1:], columns=mIndPureLig_Col_Names[1:], index=vSectors))

    # Influence Areas
    vNameSheets.append("Campo_Influencia")
    vDataSheets.append(pd.DataFrame(mInfluence, columns=vSectors, index=vSectors))

    # Hypothetical Extractions
    vNameSheets.append("Extr_Hip")
    vDataSheets.append(pd.DataFrame(mExtractions[:, 1:], columns=mExtractions_Col_Names[1:], index=vSectors))

    # Structural Decomposition
    if doStructure:
        # Results
        vNameSheets.append(f"Decomp_Estrutural_{nAno}_{nAno_StructuralDecomposition}")
        vDataSheets.append(
            pd.DataFrame(mDecomposition[:, 1:], columns=mDecomposition_Col_Names[1:], index=mDecomposition_Index)
        )

    # Direct coefficients (open model)
    vNameSheets.append("Coef_Diretos_Aberto (mA)")
    vDataSheets.append(pd.DataFrame(mA, columns=vSectors, index=vSectors))

    # Leontief (open model)
    vNameSheets.append("Leontief Aberto (mB)")
    vDataSheets.append(pd.DataFrame(mB, columns=vSectors, index=vSectors))

    ## Direct coefficients (closed model)
    # Appending necessary things to indexes/columns
    colClosed = np.append(vSectors, "Consumo das Famílias")
    indexClosed = np.append(vSectors, "Remunerações")

    # Creating dataframe
    vNameSheets.append("Coef_Diretos_Fechado (mA)")
    vDataSheets.append(pd.DataFrame(mA_closed, columns=colClosed, index=indexClosed))

    # Leontief (closed model)
    vNameSheets.append("Leontief Fechado (mB)")
    vDataSheets.append(pd.DataFrame(mB_closed, columns=colClosed, index=indexClosed))

    # Direct coefficients (supply-side model)
    vNameSheets.append("Coef_Diretos_Oferta (mA)")
    vDataSheets.append(pd.DataFrame(mA_supply, columns=vSectors, index=vSectors))

    # Leontief (supply-side model)
    vNameSheets.append("Matriz de Ghosh")
    vDataSheets.append(pd.DataFrame(mGhosh, columns=vSectors, index=vSectors))

    ## Writing Excel File to 'Output' directory
    Support.write_data_excel(sDirectory="./Output/Tabelas_Scripts_Agg/Análises/",
                             sFileName=f"Resultados_{nAno}_{nSectors}{AggregIndicator}.xlsx",
                             vSheetName=vNameSheets, vDataSheet=vDataSheets)

    ### ============================================================================================
    ### Creating Graphs (if requested)
    ### ============================================================================================

    if saveFig:
        ## Abbreviating sectors names for graph labels (if necessary)
        # If < 68 sectors, abbreviate sectors; else, use sector's numbers
        if nSectors < 68:
            vSectors_Graph = Support.abbreviate_sectors_names(vSectors)
        else:
            vSectors_Graph = np.arange(1, nSectors + 1).astype(str)

        if doStructure:
            if nSectors1 < 68:
                vSectors_Graph1 = Support.abbreviate_sectors_names(vSectors1)
            else:
                vSectors_Graph1 = np.arange(1, nSectors1 + 1).astype(str)

        print(f"Creating figures... ({datetime.datetime.now()})")

        ## Production Multipliers
        figSimpleProdMult = Support.bar_plot(
            vData=mProdMultipliers[:, 1], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Simples de Produção - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Simples_Producao_{nAno}{AggregIndicator}"
        )
        figTotalProdMult = Support.bar_plot(
            vData=mProdMultipliers[:, 2], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Produção - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Totais_Producao_{nAno}{AggregIndicator}"
        )
        figTotalTruncMult = Support.bar_plot(
            vData=mProdMultipliers[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Produção Truncados - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Tot_Prod_Trunc_{nAno}{AggregIndicator}"
        )

        ## Employment multipliers
        figSimpleEmpMult = Support.bar_plot(
            vData=mEmpMultipliers[:, 1], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Simples de Emprego - {nAno}", sXTitle="Setores",
            sCaption=f"{sCaption} Nota: por R$ milhão.",
            sFigName=f"Mults_Simples_Emprego_{nAno}{AggregIndicator}", yadjust=0.1
        )
        figType1EmpMult = Support.bar_plot(
            vData=mEmpMultipliers[:, 2], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Emprego (Tipo I) - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Tipo1_Emprego_{nAno}{AggregIndicator}", yadjust=0.05
        )
        figTotEmpMult = Support.bar_plot(
            vData=mEmpMultipliers[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Emprego (Truncados) - {nAno}", sXTitle="Setores",
            sCaption=f"{sCaption} Nota: por R$ milhão.",
            sFigName=f"Mult_Totais_Emprego_{nAno}{AggregIndicator}", yadjust=0.1
        )
        figType2EmpMult = Support.bar_plot(
            vData=mEmpMultipliers[:, 4], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Emprego (Tipo II) - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Tipo2_Emprego_{nAno}{AggregIndicator}", yadjust=0.08
        )

        ## Income multipliers
        ## Plotting all multipliers
        figSimpleIncomeMult = Support.bar_plot(
            vData=mIncomeMultipliers[:, 1], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Simples de Renda - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Simples_Renda_{nAno}{AggregIndicator}", yadjust=0.0005
        )
        figType1IncomeMult = Support.bar_plot(
            vData=mIncomeMultipliers[:, 2], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Renda (Tipo I) - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Tipo1_Renda_{nAno}{AggregIndicator}"
        )
        figTotIncomeMult = Support.bar_plot(
            vData=mIncomeMultipliers[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Renda (Truncados) - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Totais_Renda_{nAno}{AggregIndicator}", yadjust=0.0005
        )
        figType2IncomeMult = Support.bar_plot(
            vData=mIncomeMultipliers[:, 4], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Renda (Tipo II) - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Mult_Tipo2_Renda_{nAno}{AggregIndicator}", yadjust=0.002
        )

        ## Índices de Ligação
        # Normais
        figIndLig = Support.named_scatter_plot(
            x=mIndLig[:, 3], y=mIndLig[:, 1],
            inf_lim=0.5, sup_lim=1.5,
            sTitle=f"Índices de Ligação e Setores-Chave - {nAno}",
            sXTitle="Índice de Ligação para Frente  - Matriz de Ghosh", sYTitle="Índice de Ligação para Trás",
            vLabels=vSectors_Graph, sCaption=sCaption,
            sFigName=f"Ind_Lig_{nAno}{AggregIndicator}"
        )
        # Puros
        figIndPureLig = Support.named_scatter_plot(
            x=mIndPureLigNorm[:, 2], y=mIndPureLigNorm[:, 1],
            inf_lim=0, sup_lim=math.ceil(np.max(mIndPureLigNorm[:, 1:3])), nTextLimit=1,
            sTitle=f"Índices de Ligação Puros Normalizados e Setores-Chave - {nAno}",
            sXTitle="Índice Puro de Ligação para Frente Normalizados (PFLN)",
            sYTitle="Índice de Ligação para Trás Normalizados (PBLN)",
            vLabels=vSectors_Graph, sCaption=sCaption,
            sFigName=f"Ind_Lig_Puros_{nAno}{AggregIndicator}"
        )

        ## Hypotethical extraction
        # BL % (production loss if the sector doesn't buy anything from the rest of economy,
        # relative to total economic production)
        figExtractionBackwards = Support.bar_plot(
            vData=mExtractions[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Perda de Produção por Extração Hipótetica - Estrutura de Compras (%) - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Extracao_Hipotetica_Compras_{nAno}{AggregIndicator}",
            yadjust=0.01, sBarColor="green"
        )
        # FL % (production loss if the sector doesn't sell anything to the other economic sectors,
        # relative to total economic production)
        figExtractionForwards = Support.bar_plot(
            vData=mExtractions[:, 4], vXLabels=vSectors_Graph,
            sTitle=f"Perda de Produção por Extração Hipótetica - Estrutura de Vendas (%) - {nAno}", sXTitle="Setores",
            sCaption=sCaption,
            sFigName=f"Extracao_Hipotetica_Vendas_{nAno}{AggregIndicator}",
            yadjust=0.01, sBarColor="green"
        )

        ## Structural decomposition
        if doStructure:
            figDeltaX = Support.bar_plot(
                vData=mDecomposition[:nSectors1, 1], vXLabels=vSectors_Graph1,
                sTitle=f"Variação da Produção {nAno_StructuralDecomposition} - {nAno} (R$ Milhões 2010)",
                sXTitle="Setores",
                sCaption=sCaption_StrucuturalDecomposition,
                sFigName=f"Var_Prod{AggregIndicator}_{nAno_StructuralDecomposition}-{nAno}",
                sBarColor="darkblue", bAnnotate=False, nDirectory=nSectors
            )
            figDeltaTech = Support.bar_plot(
                vData=mDecomposition[:nSectors1, 2], vXLabels=vSectors_Graph1,
                sTitle=f"Decomposição - Variação Tecnológica {nAno_StructuralDecomposition} - {nAno}",
                sXTitle="Setores",
                sCaption=sCaption_StrucuturalDecomposition,
                sFigName=f"Var_Tecno{AggregIndicator}_{nAno_StructuralDecomposition}-{nAno}",
                sBarColor="darkred", bAnnotate=False, nDirectory=nSectors
            )
            figDeltaDemand = Support.bar_plot(
                vData=mDecomposition[:nSectors1, 3], vXLabels=vSectors_Graph1,
                sTitle=f"Decomposição - Variação da Demanda Final {nAno_StructuralDecomposition} - {nAno}",
                sXTitle="Setores",
                sCaption=sCaption_StrucuturalDecomposition,
                sFigName=f"Var_DemFinal{AggregIndicator}_{nAno_StructuralDecomposition}-{nAno}",
                sBarColor="dodgerblue", bAnnotate=False, nDirectory=nSectors
            )

        ## Influence matrix
        figInfluenceCont, figInfluenceDisc = \
            Support.influence_matrix_graph(mInfluence, vSectors_Graph, nSectors,
                                           sTitle=f"Campo de Influência - {nAno}",
                                           sFigName=f"Campo_de_Influencia_{nAno}{AggregIndicator}"
                                           )

    ### ============================================================================================

    ## Ending everything
    time_diff = datetime.datetime.now() - sTimeBegin
    print(f"All done! ({datetime.datetime.now()})")
    print(f"{time_diff.seconds} seconds passed.")
