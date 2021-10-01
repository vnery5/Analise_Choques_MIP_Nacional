"""
Sectorial analysis based on estimated matrices by EstimaMIP_Nacional (every version).
Based on Vale, Perobelli (2021).
Authors: João Maria de Oliveira and Vinícius de Almeida Nery Ferreira (Ipea-DF).
E-mails: joao.oliveira@ipea.gov.br and vinicius.nery@ipea.gov.br (or vnery5@gmail.com).
"""

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
    # 3: 51 sectors
    # 4: 68 sectors
    # 9: more than 68 sectors ("68+")
    # 0: other (specify number of sectors below)
    nDimension = 2

    ## Year to be analyzed
    nYear = 2018
    
    ## Use MIPs estimated under Guilhoto (2010) or Alves-Passoni, Freitas (APF) (2020)?
    bGuilhoto = True  # True or False

    ## Whether to create and save figures
    saveFig = True  # True or False

    ## Highlight one sectors? If so, which index and color?
    bHighlightSectorFigs = True  # True or False
    nIndexHighlightSectorsFigs = 3  # 3: Electricity (when using 20 sectors)
    sHighlightColor = "red"

    ## Do a structural decomposition?
    doStructure = True  # True or False
    # Year to be compared in the structural decomposition
    nYear_Decomp = 2011

    ## Open model methodology: use Guilhoto's (True) or Vale, Perobelli's (False)?
    bOpenGuilhoto = True  # True or False

    ### ============================================================================================

    ### Defining file paths and names
    if nDimension == 0:
        nSectorsFile = 67  # How many sectors?
        invalidSectorNumber = ""
    elif nDimension == 9:
        nSectorsFile = "68+"
        invalidSectorNumber = ""
    else:
        ## Dimensions -> sectors list
        listDimensions = [12, 20, 51, 68]
        try:
            nSectorsFile = listDimensions[nDimension - 1]
            invalidSectorNumber = ""
        except IndexError:
            nSectorsFile = listDimensions[3]
            invalidSectorNumber = "Couldn't find desired dimension. Running for 68 sectors."
            nDimension = 4

    ## Estimated MIPs files
    # Sheet formatting has to be the SAME as those generated by EstimaMIP (including a "difference" line at the end)
    sPathMIP = "./Input/MIPs Estimadas/"
    sAPF = "" if bGuilhoto else "_Patieene"
    sFileNameMIP = f"MIP_{nYear}_{nSectorsFile}{sAPF}.xlsx"
    sFileNameMIP_StructuralDecomposition = f"MIP_{nYear_Decomp}_{nSectorsFile}{sAPF}.xlsx"

    # Joining path and file name
    sFileMIP = sPathMIP + sFileNameMIP
    sFileMIP_StructuralDecomposition = sPathMIP + sFileNameMIP_StructuralDecomposition

    # Sheet name
    sSheetNameMIP = "MIP"

    ## Figure Indicator
    saveFigIndicator = " (WITH figures)..." if saveFig else " (WITHOUT figures)..."

    ### ============================================================================================

    ### Print start
    sTimeBegin = datetime.datetime.now()
    print("======================= INPUT OUTPUT INDICATORS - VERSION 2 =======================")
    print(f"Starting for year = {nYear} and dimension = {nDimension}{saveFigIndicator} ({sTimeBegin})")
    print(invalidSectorNumber)

    ## Read necessary matrices and get number of sectors and sector's names
    dfMIP, nSectors, vSectors, mZ, mY, mX, mC, mV, mR, mE, mSP, vAVNames, mAddedValue, vFDNames, mFinalDemand = \
        Support.read_estimated_mip(sFileMIP, sSheetName=sSheetNameMIP)

    # Getting payment sector of the final demand components
    mSP_FD = dfMIP.values[nSectors + 1:nSectors + 6, nSectors + 1:-2]

    ### ============================================================================================
    ### Technical Coeficientes (Closed, Open and Supply)
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
    ### Calculating Coefficients
    ## Methodology proposed by Vale, Perobelli, 2021
    # n = 51 sectors doesn't have the differentiation between EOB and RM (required for Guilhoto's methodology)
    if not bOpenGuilhoto or nDimension == 3:
        ## Indicator for differentiation when saving spreadsheet
        sOpenGuilhotoIndicator = "_Open_Perobelli"
        print("Open model: Vale and Perobelli")

        # Consumption: what families (not including ISFLSF) consume of each sector in respect to total income
        # (only including remunerations or, in other words, excluding EOB and RM)
        hC = mC / np.sum(mR)

        # Remunerations (transposed): percentage of each sector's production that becomes income
        hR = mR / mX
        hR = hR.T

    ## Methodology proposed by Guilhoto (
    # Idea: income = total consumption (usually < in TRUs and MIPs when working with only remuneration)
    else:
        ## Indicator for differentiation when saving spreadsheet
        sOpenGuilhotoIndicator = "_Open_Guilhoto"
        print("Open model: Guilhoto")

        ## Column and row numbers (base 0) (we don't need to worry about n = 51 because Guilhoto isn't possible)
        # Final Demand
        nColISFLSFConsumption = 2
        nColFamilyConsumption = 3
        # Added Value
        nRowRemunerations = 1
        nRowRM = 8
        nRowEOB = 9

        ## Concatenating (vertically) final demand productive sectors and imports/taxes (without total lines)
        mFinalDemand_Import_Taxes = np.vstack((mFinalDemand, mSP_FD))

        ## Finding consumption and total income under Guilhoto
        mC_Guilhoto, mR_Guilhoto, nTotalConsumption = \
            Support.open_model_guilhoto(mFinalDemand_Import_Taxes, mAddedValue, nSectors, nColISFLSFConsumption,
                                        nColFamilyConsumption, nRowRemunerations, nRowRM, nRowEOB)

        ## Calculating coefficients
        hC = mC_Guilhoto / nTotalConsumption
        hR = mR_Guilhoto / mX

    ## A and Leontief's matrix in the closed model
    """
    In this case, Leontief's matrix shows, given a one unit increase in demand for products of sector j,
    what are the direct, indirect and induced prerequisites for each sector's production.
    Therefore, the coefficients of mB_closed are larger than those of mB and their difference can be interpreted
    as the induced impacts of household consumption expansion upon the production of each sector.
    """
    mA_closed, mB_closed = Support.leontief_closed(mA, hC, hR, nSectors)

    ### Supply-side model
    mA_supply, mGhosh = Support.ghosh_supply(mZ, mX, nSectors)

    ### ============================================================================================
    ### Production Multipliers
    ### ============================================================================================

    ## Simple Production Multipliers (open model)
    """
    Given an increase of 1 in the demand of sector j, how much output/production is generated in the economy?
    """
    MP = np.sum(mB, axis=0)

    ## Total Production Multipliers (closed model)
    """
    Given an increase of 1 in the demand of sector j, how much output/production is generated 
    in the economy including the induced effects of household consumption expansion?
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

    ## Total Truncated Production Multipliers (closed model)
    """
    Given an increase of 1 in the demand of sector j, how much output is generated 
    in the economy, including the induced effects of household consumption expansion 
    (but only considering the productive sectors; in other words, not considering
    the direct impact of household consumption on GDP, but only its induced effects)?
    """
    MPTT = np.sum(mB_closed[:nSectors, :nSectors], axis=0)
    mInducedEffectTrunc = np.sum(mB_closed[:nSectors, :nSectors], axis=0) - np.sum(mB, axis=0)

    ## Creating array with all multipliers
    mProdMultipliers_Col_Names = [
        "Setor", "Multiplicador Simples de Produção", "Multiplicador Total de Produção",
        "Multiplicador Total de Produção Truncado", "Efeito Direto", "Efeito Indireto", "Efeito Induzido"
    ]
    ## Creating table with all multipliers
    mProdMultipliers = np.vstack((vSectors, MP, MPT, MPTT, mDirectEffect, mIndirectEffect, mInducedEffect)).T

    ### ============================================================================================
    ### Labor Multipliers
    """
    In line with the production multipliers, the simple labor multipliers tell us how many jobs
    are generated (directly and indirectly) when there is a 1 million unit increase in demand for sector's j products
    The total truncated labor multipliers, in turn, includes the induced effects of consumption expansion
    Type I multipliers tell us how many jobs are directly and indirectly generated for each job directly created
    Type II multipliers tell us how many jobs are directly, indirectly and "inducedly" generated for each direct job.
    """
    ### ============================================================================================

    ## Creating array with all multiplier names
    mEmpMultipliers_Col_Names = [
        "Setor", "Coeficiente", "Multiplicador Simples de Emprego", "Multiplicador de Emprego Tipo I",
        "Multiplicador Total de Emprego (truncado)", "Multiplicador de Emprego Tipo II",
        "Efeito Direto", "Efeito Indireto", "Efeito Induzido"
    ]
    ## Creating table with all multipliers
    mEmpMultipliers = Support.calc_multipliers(mE, mX, mA, mB, mB_closed, vSectors, nSectors)

    ### ============================================================================================
    ### Income Multipliers: Same interpretation as that of labor multipliers
    ### ============================================================================================

    ## Creating array with all multiplier names
    mIncomeMultipliers_Col_Names = [
        "Setor", "Coeficiente", "Multiplicador Simples de Renda", "Multiplicador de Renda Tipo I",
        "Multiplicador Total de Renda (truncado)", "Multiplicador de Renda Tipo II",
        "Efeito Direto", "Efeito Indireto", "Efeito Induzido"
    ]
    ## Creating table with all multipliers
    mIncomeMultipliers = Support.calc_multipliers(mR, mX, mA, mB, mB_closed, vSectors, nSectors)

    ### ============================================================================================
    ### Taxes Multipliers: Same interpretation as above and considering only sectorial taxes
    ### (not including any final demand components present in mSP_FD)
    ### ============================================================================================

    ## Creating array with all multiplier names
    mTaxesMultipliers_Col_Names = [
        "Setor", "Coeficiente", "Multiplicador Simples de Impostos", "Multiplicador de Impostos Tipo I",
        "Multiplicador Total de Impostos (truncado)", "Multiplicador de Impostos Tipo II",
        "Efeito Direto", "Efeito Indireto", "Efeito Induzido"
    ]

    ## "Vector" (nSectors x 1 matrix) containing all taxes paid by sectors
    mTaxes = np.sum(mSP[1:5, :], axis=0, keepdims=True).T

    ## Creating table with all multipliers
    mTaxesMultipliers = Support.calc_multipliers(mTaxes, mX, mA, mB, mB_closed, vSectors, nSectors)

    ### ============================================================================================
    ### Linkages (Hirschman-Rasmussen - HR) and Variance Coefficients
    """
    The indices show which sectors have larger chaining impacts in the economic, not only buying from other
    sectors in order to meet rises in its final demand ("backwards"/dispersion power - 
    how much the sector demands from others), but also producing to meet rising final demand in the other 
    economic sectors ("forwards"/dispersion sensibility - how much the sector is demand by the others).
    We can normalize this impacts (dividing by the indicator's mean) in order to see which sectors
    are relatively more important/chained in the economy (norm. ind. > 1 -> bigger than the mean).
    """
    ### ============================================================================================

    ## Meaning across lines (get column totals) in Leontief's matrix: backwards linkages
    # Indices show how much other sector's have to produce in order to meet a rise of 1 unit in sector's j final demand
    nBL_Bar = np.mean(mB, axis=0)

    ## Adding across columns (get row/line totals) in Leontief's matrix: forward linkages
    # Indices show how much sector i has to produce in order to meet a rise of 1 in final demand of the whole economy
    nFL_Bar = np.mean(mB, axis=1)

    ## Mean of all indirect and direct technical coefficients
    nMT = np.sum(mB) / nSectors**2

    ## Backwards-Looking Index (Uj)
    BL = nBL_Bar / nMT
    ## Forward-Looking Index (Ui)
    FL = nFL_Bar / nMT

    ## Using Ghosh's matrix (supply-side model)
    MLG = np.mean(mGhosh, axis=1)
    G_star = np.sum(mGhosh) / nSectors**2
    FLG = MLG / G_star

    ## Joining all indices
    mIndLig_Col_Names = ["Setor", "Para Trás", "Para Frente", "Para Frente Ghosh"]
    mIndLig = np.vstack((vSectors, BL, FL, FLG)).T

    ## Checking key sectors
    # Transforming into a dataframe
    dfIndLig = pd.DataFrame(mIndLig, columns=mIndLig_Col_Names[:4])
    dfIndLig['Setores-Chave'] = np.where(
        (dfIndLig['Para Trás'] >= 1) & (dfIndLig['Para Frente Ghosh'] >= 1),
        "Setor-Chave",
        "-"
    )

    # Updating array
    mIndLig_Col_Names = ["Setor", "Para Trás", "Para Frente", "Para Frente Ghosh", "Setor-Chave"]
    mIndLig = dfIndLig.to_numpy()

    ### Variance Coefficients
    ## The lower the coefficient, the larger the number of sectors impacted by that sector's...
    # ... increase in production/sales
    CVi = np.std(mB, axis=1, ddof=1) / nFL_Bar
    # ... increase in final demand
    CVj = np.std(mB, axis=0, ddof=1) / nBL_Bar

    ## Joining into an aggregated table
    mVarCoef_Col_Names = ["Setor", "CVi", "CVj"]
    mVarCoef = np.vstack((vSectors, CVi, CVj)).T

    ### Pure Linkages (GHS or, in portuguese, IPL)
    """
    As pointed out by Guilhoto (2009), the traditional indexes don't take into consideration
    the production levels of each sector. The "pure" or "generalized" version doesn't have this problem.
    The backwards pure index (PBL) shows the impact of the production value of sector j upon the rest of the economy,
    excluding self-demand for its own inputs and the returns of the rest of the economy to the sector, representing,
    therefore, the relative importance of that sector's DEMAND.
    The forward index (PFL) indicates the impact of the production value of the rest of the economy upon the
    production of sector j, representing, therefore, the relative importance of that sector's SUPPLY.
    """

    ## Calculating IPL
    mIndPureLig_Col_Names = ["Setor", "PBL", "PFL", "PTL"]
    mIndPureLig, mIndPureLigNorm = Support.calc_ipl(mY, mA, vSectors, nSectors)

    ## Checking key sectors
    # Transforming into a dataframe
    dfIndPureLigNorm = pd.DataFrame(mIndPureLigNorm, columns=mIndPureLig_Col_Names)
    dfIndPureLigNorm['Setores-Chave'] = np.where(dfIndPureLigNorm['PTL'] >= 1, "Setor-Chave", "-")

    # Updating array
    mIndPureLig_Col_Names = ["Setor", "PBL", "PFL", "PTL", "Setor-Chave"]
    mIndPureLigNorm = dfIndPureLigNorm.to_numpy()

    ### ======================================================================================
    ### Influence Matrices
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
    nIncrement = 0.001

    ### Influence Matrix (see Vale, Perobelli, p. 98-103)
    mInfluence = Support.influence_matrix(mA, nIncrement, nSectors)

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
        # Disaggregation/aggregation for nSectors other than 12, 20, 51, 68 is not yet supported :/
        # Year 1 (= as the rest of the analysis)
        dfMIP1, nSectors1, vSectors1, mZ1, mY1, mX1, mC1, mV1, mR1, mE1, mSP1, \
            vAVNames1, mAddedValue1, vFDNames1, mFinalDemand1 = \
            Support.read_estimated_mip(sFileMIP, sSheetName=sSheetNameMIP)

        # Year 0
        dfMIP0, nSectors0, vSectors0, mZ0, mY0, mX0, mC0, mV0, mR0, mE0, mSP0, \
            vAVNames0, mAddedValue0, vFDNames0, mFinalDemand0 = \
            Support.read_estimated_mip(sFileMIP_StructuralDecomposition, sSheetName=sSheetNameMIP)

        ### Inflating year 0 prices to year 1's
        ## Getting price indexes (2010 = 100)
        mZ_index1, mY_index1, mX_index1 = Support.read_deflator(nYear, nSectors1)
        mZ_index0, mY_index0, mX_index0 = Support.read_deflator(nYear_Decomp, nSectors0)

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
                                    f"Índices de Preços da Produção Total {nYear}",
                                    f"Índices de Preços da Produção Total {nYear_Decomp}"]
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

    # Original Input-Output matrix
    vNameSheets = ["MIP_Original"]
    vDataSheets = [dfMIP]

    # Production Multipliers
    vNameSheets.append("Mult_Prod")
    vDataSheets.append(pd.DataFrame(mProdMultipliers[:, 1:], columns=mProdMultipliers_Col_Names[1:], index=vSectors))

    # Employment/Labor Multipliers
    vNameSheets.append("Mult_Trab")
    vDataSheets.append(pd.DataFrame(mEmpMultipliers[:, 1:], columns=mEmpMultipliers_Col_Names[1:], index=vSectors))

    # Income Multipliers
    vNameSheets.append("Mult_Renda")
    vDataSheets.append(
        pd.DataFrame(mIncomeMultipliers[:, 1:], columns=mIncomeMultipliers_Col_Names[1:], index=vSectors)
    )

    # Taxes Multipliers
    vNameSheets.append("Mult_Imp")
    vDataSheets.append(
        pd.DataFrame(mTaxesMultipliers[:, 1:], columns=mTaxesMultipliers_Col_Names[1:], index=vSectors)
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
        vNameSheets.append(f"Decomp_Estrutural_{nYear}_{nYear_Decomp}")
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
    Support.write_data_excel(sDirectory="./Output/Tabelas_Main/Análises/",
                             sFileName=f"Resultados_{nYear}_{nSectors}{sAPF}{sOpenGuilhotoIndicator}.xlsx",
                             vSheetName=vNameSheets, vDataSheet=vDataSheets)

    ### ============================================================================================
    ### Creating Graphs (if requested)
    ### ============================================================================================

    if saveFig:
        ## Creating color list and highlighting desired sector (if necessary)
        lColours = ["#595959"] * nSectors
        if bHighlightSectorFigs:
            lColours[nIndexHighlightSectorsFigs] = sHighlightColor

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
        Support.bar_plot(
            vData=mProdMultipliers[:, 1], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Simples de Produção - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Prod_Simples_{nYear}", BarColor=lColours
        )
        Support.bar_plot(
            vData=mProdMultipliers[:, 2], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Produção - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Prod_Totais_{nYear}", BarColor=lColours
        )
        Support.bar_plot(
            vData=mProdMultipliers[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Produção Truncados - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Prod_TotTrunc_{nYear}", BarColor=lColours
        )

        ## Employment Multipliers
        Support.bar_plot(
            vData=mEmpMultipliers[:, 1], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Simples de Emprego - {nYear}", sXTitle="Setores", BarColor=lColours,
            sFigName=f"Mult_Emp_Simples_{nYear}", nY_Adjust=0.1
        )
        Support.bar_plot(
            vData=mEmpMultipliers[:, 2], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Emprego (Tipo I) - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Emp_Tipo1_{nYear}", nY_Adjust=0.05, BarColor=lColours
        )
        Support.bar_plot(
            vData=mEmpMultipliers[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Emprego (Truncados) - {nYear}", sXTitle="Setores", BarColor=lColours,
            sFigName=f"Mult_Emp_Tot_{nYear}", nY_Adjust=0.1
        )
        Support.bar_plot(
            vData=mEmpMultipliers[:, 4], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Emprego (Tipo II) - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Emp_Tipo2_{nYear}", nY_Adjust=0.08, BarColor=lColours
        )

        ## Income Multipliers
        Support.bar_plot(
            vData=mIncomeMultipliers[:, 1], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Simples de Renda - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Renda_Simples_{nYear}", nY_Adjust=0.0005, BarColor=lColours
        )
        Support.bar_plot(
            vData=mIncomeMultipliers[:, 2], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Renda (Tipo I) - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Renda_Tipo1_{nYear}", BarColor=lColours
        )
        Support.bar_plot(
            vData=mIncomeMultipliers[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Renda (Truncados) - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Renda_Tot_{nYear}", nY_Adjust=0.0005, BarColor=lColours
        )
        Support.bar_plot(
            vData=mIncomeMultipliers[:, 4], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Renda (Tipo II) - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Renda_Tipo2_{nYear}", nY_Adjust=0.002, BarColor=lColours
        )

        ## Taxes Multipliers
        Support.bar_plot(
            vData=mTaxesMultipliers[:, 1], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Simples de Impostos - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Imp_Simples_{nYear}", nY_Adjust=0.0005, BarColor=lColours
        )
        Support.bar_plot(
            vData=mTaxesMultipliers[:, 2], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Impostos (Tipo I) - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Imp_Tipo1_{nYear}", BarColor=lColours
        )
        Support.bar_plot(
            vData=mTaxesMultipliers[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores Totais de Impostos (Truncados) - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Imp_Tot_{nYear}", nY_Adjust=0.0005, BarColor=lColours
        )
        Support.bar_plot(
            vData=mTaxesMultipliers[:, 4], vXLabels=vSectors_Graph,
            sTitle=f"Multiplicadores de Impostos (Tipo II) - {nYear}", sXTitle="Setores",
            sFigName=f"Mult_Imp_Tipo2_{nYear}", nY_Adjust=0.002, BarColor=lColours
        )

        ## Linkages
        # Traditional (HR)
        Support.named_scatter_plot(
            x=mIndLig[:, 3], y=mIndLig[:, 1],
            inf_lim=0.5, sup_lim=1.5,
            sTitle=f"Índices de Ligação e Setores-Chave - {nYear}",
            sXTitle="Índice de Ligação para Frente  - Matriz de Ghosh", sYTitle="Índice de Ligação para Trás",
            vLabels=vSectors_Graph,  sFigName=f"Ind_Lig_{nYear}", PointColor=lColours
        )
        # Pure (GHS)
        Support.named_scatter_plot(
            x=mIndPureLigNorm[:, 2], y=mIndPureLigNorm[:, 1],
            inf_lim=0, sup_lim=math.ceil(np.max(mIndPureLigNorm[:, 1:3])), nTextLimit=1,
            sTitle=f"Índices de Ligação Puros Normalizados e Setores-Chave - {nYear}",
            sXTitle="Índice Puro de Ligação para Frente Normalizados (PFLN)",
            sYTitle="Índice de Ligação para Trás Normalizados (PBLN)",
            vLabels=vSectors_Graph,  sFigName=f"Ind_Lig_Puros_{nYear}", PointColor=lColours
        )

        ## Hypothetical extraction
        # BL % (production loss if the sector doesn't buy anything from the rest of economy,
        # relative to total economic production)
        Support.bar_plot(
            vData=mExtractions[:, 3], vXLabels=vSectors_Graph,
            sTitle=f"Perda de Produção por Extração Hipótetica - Estrutura de Compras (%) - {nYear}", sXTitle="Setores",
            sFigName=f"Extr_Hipo_Compras_{nYear}", nY_Adjust=0.01, BarColor="green"
        )
        # FL % (production loss if the sector doesn't sell anything to the other economic sectors,
        # relative to total economic production)
        Support.bar_plot(
            vData=mExtractions[:, 4], vXLabels=vSectors_Graph,
            sTitle=f"Perda de Produção por Extração Hipótetica - Estrutura de Vendas (%) - {nYear}", sXTitle="Setores",
            sFigName=f"Extr_Hipo_Vendas_{nYear}", nY_Adjust=0.01, BarColor="green"
        )

        ## Influence matrix
        Support.influence_matrix_graph(mInfluence, vSectors_Graph, nSectors,
                                       sTitle=f"Campo de Influência - {nYear}",
                                       sFigName=f"Campo_de_Influencia_{nYear}"
                                       )
        ## Structural decomposition
        if doStructure:
            Support.bar_plot(
                vData=mDecomposition[:nSectors1, 1], vXLabels=vSectors_Graph1,
                sTitle=f"Variação da Produção {nYear_Decomp} - {nYear} (R$ Milhões 2010)",
                sXTitle="Setores", sFigName=f"Var_Prod_{nYear_Decomp}-{nYear}",
                BarColor="darkblue", bAnnotate=False, nDirectory=nSectors
            )
            Support.bar_plot(
                vData=mDecomposition[:nSectors1, 2], vXLabels=vSectors_Graph1,
                sTitle=f"Decomposição - Variação Tecnológica {nYear_Decomp} - {nYear}",
                sXTitle="Setores", sFigName=f"Var_Tecno_{nYear_Decomp}-{nYear}",
                BarColor="darkred", bAnnotate=False, nDirectory=nSectors
            )
            Support.bar_plot(
                vData=mDecomposition[:nSectors1, 3], vXLabels=vSectors_Graph1,
                sTitle=f"Decomposição - Variação da Demanda Final {nYear_Decomp} - {nYear}",
                sXTitle="Setores", sFigName=f"Var_DemFinal_{nYear_Decomp}-{nYear}",
                BarColor="dodgerblue", bAnnotate=False, nDirectory=nSectors
            )

    ### ============================================================================================

    ## Ending everything
    time_diff = datetime.datetime.now() - sTimeBegin
    print(f"All done! ({datetime.datetime.now()})")
    print(f"{time_diff.seconds} seconds passed.")
