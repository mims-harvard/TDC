import numpy as np
import os

try:
    from rdkit import Chem, DataStructs
    from rdkit import rdBase

    rdBase.DisableLog("rdApp.error")
except:
    raise ImportError(
        "Please install rdkit by 'conda install -c conda-forge rdkit'! ")

try:
    import networkx as nx
except:
    raise ImportError("Please install networkx by 'pip install networkx'! ")

from ...utils import print_sys, install
from ._smartsPatts import smartsPatts

PubChemKeys = None


def InitKeys(keyList, keyDict):
    assert len(keyList) == len(keyDict.keys()), "length mismatch"
    for key in keyDict.keys():
        patt, count = keyDict[key]
        if patt != "?":
            sma = Chem.MolFromSmarts(patt)
            if not sma:
                print_sys("SMARTS parser error for key #%d: %s" % (key, patt))
            else:
                keyList[key - 1] = sma, count


def calcPubChemFingerPart1(mol, **kwargs):
    global PubChemKeys
    if PubChemKeys is None:
        PubChemKeys = [(None, 0)] * len(smartsPatts.keys())

        InitKeys(PubChemKeys, smartsPatts)
    ctor = kwargs.get("ctor", DataStructs.SparseBitVect)

    res = ctor(len(PubChemKeys) + 1)
    for i, (patt, count) in enumerate(PubChemKeys):
        if patt is not None:
            if count == 0:
                res[i + 1] = mol.HasSubstructMatch(patt)
            else:
                matches = mol.GetSubstructMatches(patt)
                if len(matches) > count:
                    res[i + 1] = 1
    return res


def func_1(mol, bits):
    ringSize = []
    temp = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    AllRingsAtom = mol.GetRingInfo().AtomRings()
    for ring in AllRingsAtom:
        ringSize.append(len(ring))
        for k, v in temp.items():
            if len(ring) == k:
                temp[k] += 1
    if temp[3] >= 2:
        bits[0] = 1
        bits[7] = 1
    elif temp[3] == 1:
        bits[0] = 1
    else:
        pass
    if temp[4] >= 2:
        bits[14] = 1
        bits[21] = 1
    elif temp[4] == 1:
        bits[14] = 1
    else:
        pass
    if temp[5] >= 5:
        bits[28] = 1
        bits[35] = 1
        bits[42] = 1
        bits[49] = 1
        bits[56] = 1
    elif temp[5] == 4:
        bits[28] = 1
        bits[35] = 1
        bits[42] = 1
        bits[49] = 1
    elif temp[5] == 3:
        bits[28] = 1
        bits[35] = 1
        bits[42] = 1
    elif temp[5] == 2:
        bits[28] = 1
        bits[35] = 1
    elif temp[5] == 1:
        bits[28] = 1
    else:
        pass
    if temp[6] >= 5:
        bits[63] = 1
        bits[70] = 1
        bits[77] = 1
        bits[84] = 1
        bits[91] = 1
    elif temp[6] == 4:
        bits[63] = 1
        bits[70] = 1
        bits[77] = 1
        bits[84] = 1
    elif temp[6] == 3:
        bits[63] = 1
        bits[70] = 1
        bits[77] = 1
    elif temp[6] == 2:
        bits[63] = 1
        bits[70] = 1
    elif temp[6] == 1:
        bits[63] = 1
    else:
        pass
    if temp[7] >= 2:
        bits[98] = 1
        bits[105] = 1
    elif temp[7] == 1:
        bits[98] = 1
    else:
        pass
    if temp[8] >= 2:
        bits[112] = 1
        bits[119] = 1
    elif temp[8] == 1:
        bits[112] = 1
    else:
        pass
    if temp[9] >= 1:
        bits[126] = 1
    else:
        pass
    if temp[10] >= 1:
        bits[133] = 1
    else:
        pass

    return ringSize, bits


def func_2(mol, bits):
    """*Internal Use Only*
    saturated or aromatic carbon-only ring
    """
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize = []
    temp = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "SINGLE":
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
        ######## aromatic carbon-only
        aromatic = True
        AllCarb = True
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "AROMATIC":
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() != 6 or EndAtom.GetAtomicNum() != 6:
                AllCarb = False
                break
        if aromatic == True and AllCarb == True:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
    if temp[3] >= 2:
        bits[1] = 1
        bits[8] = 1
    elif temp[3] == 1:
        bits[1] = 1
    else:
        pass
    if temp[4] >= 2:
        bits[15] = 1
        bits[22] = 1
    elif temp[4] == 1:
        bits[15] = 1
    else:
        pass
    if temp[5] >= 5:
        bits[29] = 1
        bits[36] = 1
        bits[43] = 1
        bits[50] = 1
        bits[57] = 1
    elif temp[5] == 4:
        bits[29] = 1
        bits[36] = 1
        bits[43] = 1
        bits[50] = 1
    elif temp[5] == 3:
        bits[29] = 1
        bits[36] = 1
        bits[43] = 1
    elif temp[5] == 2:
        bits[29] = 1
        bits[36] = 1
    elif temp[5] == 1:
        bits[29] = 1
    else:
        pass
    if temp[6] >= 5:
        bits[64] = 1
        bits[71] = 1
        bits[78] = 1
        bits[85] = 1
        bits[92] = 1
    elif temp[6] == 4:
        bits[64] = 1
        bits[71] = 1
        bits[78] = 1
        bits[85] = 1
    elif temp[6] == 3:
        bits[64] = 1
        bits[71] = 1
        bits[78] = 1
    elif temp[6] == 2:
        bits[64] = 1
        bits[71] = 1
    elif temp[6] == 1:
        bits[64] = 1
    else:
        pass
    if temp[7] >= 2:
        bits[99] = 1
        bits[106] = 1
    elif temp[7] == 1:
        bits[99] = 1
    else:
        pass
    if temp[8] >= 2:
        bits[113] = 1
        bits[120] = 1
    elif temp[8] == 1:
        bits[113] = 1
    else:
        pass
    if temp[9] >= 1:
        bits[127] = 1
    else:
        pass
    if temp[10] >= 1:
        bits[134] = 1
    else:
        pass
    return ringSize, bits


def func_3(mol, bits):
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize = []
    temp = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "SINGLE":
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
        ######## aromatic nitrogen-containing
        aromatic = True
        ContainNitro = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "AROMATIC":
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() == 7 or EndAtom.GetAtomicNum() == 7:
                ContainNitro = True
                break
        if aromatic == True and ContainNitro == True:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
    if temp[3] >= 2:
        bits[2] = 1
        bits[9] = 1
    elif temp[3] == 1:
        bits[2] = 1
    else:
        pass
    if temp[4] >= 2:
        bits[16] = 1
        bits[23] = 1
    elif temp[4] == 1:
        bits[16] = 1
    else:
        pass
    if temp[5] >= 5:
        bits[30] = 1
        bits[37] = 1
        bits[44] = 1
        bits[51] = 1
        bits[58] = 1
    elif temp[5] == 4:
        bits[30] = 1
        bits[37] = 1
        bits[44] = 1
        bits[51] = 1
    elif temp[5] == 3:
        bits[30] = 1
        bits[37] = 1
        bits[44] = 1
    elif temp[5] == 2:
        bits[30] = 1
        bits[37] = 1
    elif temp[5] == 1:
        bits[30] = 1
    else:
        pass
    if temp[6] >= 5:
        bits[65] = 1
        bits[72] = 1
        bits[79] = 1
        bits[86] = 1
        bits[93] = 1
    elif temp[6] == 4:
        bits[65] = 1
        bits[72] = 1
        bits[79] = 1
        bits[86] = 1
    elif temp[6] == 3:
        bits[65] = 1
        bits[72] = 1
        bits[79] = 1
    elif temp[6] == 2:
        bits[65] = 1
        bits[72] = 1
    elif temp[6] == 1:
        bits[65] = 1
    else:
        pass
    if temp[7] >= 2:
        bits[100] = 1
        bits[107] = 1
    elif temp[7] == 1:
        bits[100] = 1
    else:
        pass
    if temp[8] >= 2:
        bits[114] = 1
        bits[121] = 1
    elif temp[8] == 1:
        bits[114] = 1
    else:
        pass
    if temp[9] >= 1:
        bits[128] = 1
    else:
        pass
    if temp[10] >= 1:
        bits[135] = 1
    else:
        pass
    return ringSize, bits


def func_4(mol, bits):
    AllRingsBond = mol.GetRingInfo().BondRings()
    ringSize = []
    temp = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for ring in AllRingsBond:
        ######### saturated
        nonsingle = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "SINGLE":
                nonsingle = True
                break
        if nonsingle == False:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
        ######## aromatic heteroatom-containing
        aromatic = True
        heteroatom = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "AROMATIC":
                aromatic = False
                break
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [
                    1, 6
            ] or EndAtom.GetAtomicNum() not in [
                    1,
                    6,
            ]:
                heteroatom = True
                break
        if aromatic == True and heteroatom == True:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
    if temp[3] >= 2:
        bits[3] = 1
        bits[10] = 1
    elif temp[3] == 1:
        bits[3] = 1
    else:
        pass
    if temp[4] >= 2:
        bits[17] = 1
        bits[24] = 1
    elif temp[4] == 1:
        bits[17] = 1
    else:
        pass
    if temp[5] >= 5:
        bits[31] = 1
        bits[38] = 1
        bits[45] = 1
        bits[52] = 1
        bits[59] = 1
    elif temp[5] == 4:
        bits[31] = 1
        bits[38] = 1
        bits[45] = 1
        bits[52] = 1
    elif temp[5] == 3:
        bits[31] = 1
        bits[38] = 1
        bits[45] = 1
    elif temp[5] == 2:
        bits[31] = 1
        bits[38] = 1
    elif temp[5] == 1:
        bits[31] = 1
    else:
        pass
    if temp[6] >= 5:
        bits[66] = 1
        bits[73] = 1
        bits[80] = 1
        bits[87] = 1
        bits[94] = 1
    elif temp[6] == 4:
        bits[66] = 1
        bits[73] = 1
        bits[80] = 1
        bits[87] = 1
    elif temp[6] == 3:
        bits[66] = 1
        bits[73] = 1
        bits[80] = 1
    elif temp[6] == 2:
        bits[66] = 1
        bits[73] = 1
    elif temp[6] == 1:
        bits[66] = 1
    else:
        pass
    if temp[7] >= 2:
        bits[101] = 1
        bits[108] = 1
    elif temp[7] == 1:
        bits[101] = 1
    else:
        pass
    if temp[8] >= 2:
        bits[115] = 1
        bits[122] = 1
    elif temp[8] == 1:
        bits[115] = 1
    else:
        pass
    if temp[9] >= 1:
        bits[129] = 1
    else:
        pass
    if temp[10] >= 1:
        bits[136] = 1
    else:
        pass
    return ringSize, bits


def func_5(mol, bits):
    ringSize = []
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        Allcarb = True
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "SINGLE":
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name == "AROMATIC":
                nonaromatic = False
                break
        ######## allcarb
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() != 6 or EndAtom.GetAtomicNum() != 6:
                Allcarb = False
                break
        if unsaturated == True and nonaromatic == True and Allcarb == True:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
    if temp[3] >= 2:
        bits[4] = 1
        bits[11] = 1
    elif temp[3] == 1:
        bits[4] = 1
    else:
        pass
    if temp[4] >= 2:
        bits[18] = 1
        bits[25] = 1
    elif temp[4] == 1:
        bits[18] = 1
    else:
        pass
    if temp[5] >= 5:
        bits[32] = 1
        bits[39] = 1
        bits[46] = 1
        bits[53] = 1
        bits[60] = 1
    elif temp[5] == 4:
        bits[32] = 1
        bits[39] = 1
        bits[46] = 1
        bits[53] = 1
    elif temp[5] == 3:
        bits[32] = 1
        bits[39] = 1
        bits[46] = 1
    elif temp[5] == 2:
        bits[32] = 1
        bits[39] = 1
    elif temp[5] == 1:
        bits[32] = 1
    else:
        pass
    if temp[6] >= 5:
        bits[67] = 1
        bits[74] = 1
        bits[81] = 1
        bits[88] = 1
        bits[95] = 1
    elif temp[6] == 4:
        bits[67] = 1
        bits[74] = 1
        bits[81] = 1
        bits[88] = 1
    elif temp[6] == 3:
        bits[67] = 1
        bits[74] = 1
        bits[81] = 1
    elif temp[6] == 2:
        bits[67] = 1
        bits[74] = 1
    elif temp[6] == 1:
        bits[67] = 1
    else:
        pass
    if temp[7] >= 2:
        bits[102] = 1
        bits[109] = 1
    elif temp[7] == 1:
        bits[102] = 1
    else:
        pass
    if temp[8] >= 2:
        bits[116] = 1
        bits[123] = 1
    elif temp[8] == 1:
        bits[116] = 1
    else:
        pass
    if temp[9] >= 1:
        bits[130] = 1
    else:
        pass
    if temp[10] >= 1:
        bits[137] = 1
    else:
        pass
    return ringSize, bits


def func_6(mol, bits):
    ringSize = []
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        ContainNitro = False
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "SINGLE":
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name == "AROMATIC":
                nonaromatic = False
                break
        ######## nitrogen-containing
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() == 7 or EndAtom.GetAtomicNum() == 7:
                ContainNitro = True
                break
        if unsaturated == True and nonaromatic == True and ContainNitro == True:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
    if temp[3] >= 2:
        bits[5] = 1
        bits[12] = 1
    elif temp[3] == 1:
        bits[5] = 1
    else:
        pass
    if temp[4] >= 2:
        bits[19] = 1
        bits[26] = 1
    elif temp[4] == 1:
        bits[19] = 1
    else:
        pass
    if temp[5] >= 5:
        bits[33] = 1
        bits[40] = 1
        bits[47] = 1
        bits[54] = 1
        bits[61] = 1
    elif temp[5] == 4:
        bits[33] = 1
        bits[40] = 1
        bits[47] = 1
        bits[54] = 1
    elif temp[5] == 3:
        bits[33] = 1
        bits[40] = 1
        bits[47] = 1
    elif temp[5] == 2:
        bits[33] = 1
        bits[40] = 1
    elif temp[5] == 1:
        bits[33] = 1
    else:
        pass
    if temp[6] >= 5:
        bits[68] = 1
        bits[75] = 1
        bits[82] = 1
        bits[89] = 1
        bits[96] = 1
    elif temp[6] == 4:
        bits[68] = 1
        bits[75] = 1
        bits[82] = 1
        bits[89] = 1
    elif temp[6] == 3:
        bits[68] = 1
        bits[75] = 1
        bits[82] = 1
    elif temp[6] == 2:
        bits[68] = 1
        bits[75] = 1
    elif temp[6] == 1:
        bits[68] = 1
    else:
        pass
    if temp[7] >= 2:
        bits[103] = 1
        bits[110] = 1
    elif temp[7] == 1:
        bits[103] = 1
    else:
        pass
    if temp[8] >= 2:
        bits[117] = 1
        bits[124] = 1
    elif temp[8] == 1:
        bits[117] = 1
    else:
        pass
    if temp[9] >= 1:
        bits[131] = 1
    else:
        pass
    if temp[10] >= 1:
        bits[138] = 1
    else:
        pass
    return ringSize, bits


def func_7(mol, bits):

    ringSize = []
    AllRingsBond = mol.GetRingInfo().BondRings()
    temp = {3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    for ring in AllRingsBond:
        unsaturated = False
        nonaromatic = True
        heteroatom = False
        ######### unsaturated
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "SINGLE":
                unsaturated = True
                break
        ######## non-aromatic
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name == "AROMATIC":
                nonaromatic = False
                break
        ######## heteroatom-containing
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [
                    1, 6
            ] or EndAtom.GetAtomicNum() not in [
                    1,
                    6,
            ]:
                heteroatom = True
                break
        if unsaturated == True and nonaromatic == True and heteroatom == True:
            ringSize.append(len(ring))
            for k, v in temp.items():
                if len(ring) == k:
                    temp[k] += 1
    if temp[3] >= 2:
        bits[6] = 1
        bits[13] = 1
    elif temp[3] == 1:
        bits[6] = 1
    else:
        pass
    if temp[4] >= 2:
        bits[20] = 1
        bits[27] = 1
    elif temp[4] == 1:
        bits[20] = 1
    else:
        pass
    if temp[5] >= 5:
        bits[34] = 1
        bits[41] = 1
        bits[48] = 1
        bits[55] = 1
        bits[62] = 1
    elif temp[5] == 4:
        bits[34] = 1
        bits[41] = 1
        bits[48] = 1
        bits[55] = 1
    elif temp[5] == 3:
        bits[34] = 1
        bits[41] = 1
        bits[48] = 1
    elif temp[5] == 2:
        bits[34] = 1
        bits[41] = 1
    elif temp[5] == 1:
        bits[34] = 1
    else:
        pass
    if temp[6] >= 5:
        bits[69] = 1
        bits[76] = 1
        bits[83] = 1
        bits[90] = 1
        bits[97] = 1
    elif temp[6] == 4:
        bits[69] = 1
        bits[76] = 1
        bits[83] = 1
        bits[90] = 1
    elif temp[6] == 3:
        bits[69] = 1
        bits[76] = 1
        bits[83] = 1
    elif temp[6] == 2:
        bits[69] = 1
        bits[76] = 1
    elif temp[6] == 1:
        bits[69] = 1
    else:
        pass
    if temp[7] >= 2:
        bits[104] = 1
        bits[111] = 1
    elif temp[7] == 1:
        bits[104] = 1
    else:
        pass
    if temp[8] >= 2:
        bits[118] = 1
        bits[125] = 1
    elif temp[8] == 1:
        bits[118] = 1
    else:
        pass
    if temp[9] >= 1:
        bits[132] = 1
    else:
        pass
    if temp[10] >= 1:
        bits[139] = 1
    else:
        pass
    return ringSize, bits


def func_8(mol, bits):

    AllRingsBond = mol.GetRingInfo().BondRings()
    temp = {"aromatic": 0, "heteroatom": 0}
    for ring in AllRingsBond:
        aromatic = True
        heteroatom = False
        for bondIdx in ring:
            if mol.GetBondWithIdx(bondIdx).GetBondType().name != "AROMATIC":
                aromatic = False
                break
        if aromatic == True:
            temp["aromatic"] += 1
        for bondIdx in ring:
            BeginAtom = mol.GetBondWithIdx(bondIdx).GetBeginAtom()
            EndAtom = mol.GetBondWithIdx(bondIdx).GetEndAtom()
            if BeginAtom.GetAtomicNum() not in [
                    1, 6
            ] or EndAtom.GetAtomicNum() not in [
                    1,
                    6,
            ]:
                heteroatom = True
                break
        if heteroatom == True:
            temp["heteroatom"] += 1
    if temp["aromatic"] >= 4:
        bits[140] = 1
        bits[142] = 1
        bits[144] = 1
        bits[146] = 1
    elif temp["aromatic"] == 3:
        bits[140] = 1
        bits[142] = 1
        bits[144] = 1
    elif temp["aromatic"] == 2:
        bits[140] = 1
        bits[142] = 1
    elif temp["aromatic"] == 1:
        bits[140] = 1
    else:
        pass
    if temp["aromatic"] >= 4 and temp["heteroatom"] >= 4:
        bits[141] = 1
        bits[143] = 1
        bits[145] = 1
        bits[147] = 1
    elif temp["aromatic"] == 3 and temp["heteroatom"] == 3:
        bits[141] = 1
        bits[143] = 1
        bits[145] = 1
    elif temp["aromatic"] == 2 and temp["heteroatom"] == 2:
        bits[141] = 1
        bits[143] = 1
    elif temp["aromatic"] == 1 and temp["heteroatom"] == 1:
        bits[141] = 1
    else:
        pass
    return bits


def calcPubChemFingerPart2(mol):  # 116-263

    bits = [0] * 148
    bits = func_1(mol, bits)[1]
    bits = func_2(mol, bits)[1]
    bits = func_3(mol, bits)[1]
    bits = func_4(mol, bits)[1]
    bits = func_5(mol, bits)[1]
    bits = func_6(mol, bits)[1]
    bits = func_7(mol, bits)[1]
    bits = func_8(mol, bits)

    return bits


def calcPubChemFingerAll(s):
    mol = Chem.MolFromSmiles(s)
    AllBits = [0] * 881
    res1 = list(calcPubChemFingerPart1(mol).ToBitString())
    for index, item in enumerate(res1[1:116]):
        if item == "1":
            AllBits[index] = 1
    for index2, item2 in enumerate(res1[116:734]):
        if item2 == "1":
            AllBits[index2 + 115 + 148] = 1
    res2 = calcPubChemFingerPart2(mol)
    for index3, item3 in enumerate(res2):
        if item3 == 1:
            AllBits[index3 + 115] = 1
    return np.array(AllBits)


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None


def smiles2pubchem(s):
    s = canonicalize(s)
    try:
        features = calcPubChemFingerAll(s)
    except:
        print("pubchem fingerprint not working for smiles: " + s +
              " convert to 0 vectors")
        features = np.zeros((881,))
    return np.array(features)
