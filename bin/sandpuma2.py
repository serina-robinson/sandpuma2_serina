# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" SANDPUMA. """

import logging
import os, sys
import re
from typing import Any, Dict, List, Optional, Tuple

# sys.path.append(os.path.abspath('/home/mchevrette/git/antismash5/antismash')) ## just for testing, insert antismash dir here
sys.path.append(os.path.abspath('/Users/robi0916/Documents/Wageningen_UR/github/as5'))

import subprocessing, pplacer, fasta, module_results
## NOTE, above will need to be removed and
## below will likely need to be uncommented for antismash integration, but for now I've
## put these in this repository so that you can see the functions I've written (and the
## antismash dev team might has better places to put these functions now).
#from antismash.common import fasta, subprocessing, pplacer, module_results
#from antismash.common.secmet import Record

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
from Bio import Phylo
import multiprocessing
from collections import OrderedDict


class PredicatResults(module_results.ModuleResults):
    """ Results for prediCAT """
    def __init__(self, monophyly: str, forced: str, snn_score: float) -> None:
        self.monophyly = str(monophyly)
        self.forced = str(forced)
        self.snn_score = float(snn_score)

class NNResults(module_results.ModuleResults):
    """ Results for Neural network """
    def __init__(self, first_pred: str, first_prob: float, second_pred: str, second_prob: float, third_pred: str, third_prob: float) -> None:
        self.first_pred = str(first_pred)
        self.first_prob = float(first_prob)
        self.second_pred = str(second_pred)
        self.second_prob = float(second_prob)
        self.third_pred = str(third_pred)
        self.third_prob = float(third_prob)
        
class SandpumaResults(module_results.ModuleResults):
    """ Results for SANDPUMA """
    def __init__(self, predicat: PredicatResults, asm: str, pid: float, neuralnet: NNResults) -> None:
        self.predicat = predicat
        self.asm = str(asm)
        self.pid = float(pid)
        self.neuralnet = neuralnet

        
def get_parent(tree, child_clade) -> Phylo.BaseTree.TreeElement:
    ''' Given a tree and a node, returns the parental node '''
    node_path = tree.get_path(child_clade)
    if len(node_path) < 2:
        return None
    return node_path[-2]


def get_child_leaves(tree, parent) -> List:
    ''' Given a tree and a node, returns all children nodes '''
    child_leaves = []
    for leaf in tree.get_terminals():
        for node in tree.get_path(leaf):
            if(node == parent):
                child_leaves.append(leaf)
    return child_leaves


def calcscore(scaleto, distance) -> float:
    ''' Scale distance to score from 0 to 1 '''
    if(distance >= scaleto):
        return 0
    else:
        return float(scaleto - distance) / scaleto


def getscore(scaleto, nd, dist2q, leaf, o) -> float:
    ''' Calculate the SNN '''
    score = 0
    nnspec = leaf[o[0]]['spec']
    for n in o:
        curspec = leaf[o[0]]['spec']
        if(nnspec == curspec):
            tmpscore = calcscore(scaleto, float(dist2q[n]) / nd)
            if(tmpscore > 0):
                score += tmpscore
            else:
                break
        else:
            break
    return score    


def deeperdive(query: int, tree: Phylo.BaseTree, nearest1: int, nearest2: int, l: Dict[int, Dict[str, Any]])-> [str, str, str]:
    """ deeper substrate prediction triggered for non-monophyletic seqs
    Arguments:
        query: index for the query
        tree: tree
        nearest1: index for the nearest neighbor
        nearest2, index for the second nearest neighbor
        l: dictionary of leaf index to str to any
            includes group (str), id (str), spec (str), node (Phylo.node)

    Returns:
        monophyly specificity (str), hit name (str), forced call specificity (str)
    """
    ## Want query to nearest dist to be less than nearest1 to nearest2 dist
    query_to_nn1 = tree.distance(l[query]['node'], l[nearest1]['node'])
    nn1_to_nn2 = tree.distance(l[nearest1]['node'], l[nearest2]['node'])
    query_to_nn2 = tree.distance(l[query]['node'], l[nearest2]['node'])
    if((query_to_nn1 < nn1_to_nn2) and (l[nearest1]['spec'] == l[nearest2]['spec'])):
        return (l[nearest1]['spec'], l[nearest1]['id'], 'no_force_needed')
    elif((query_to_nn1 == query_to_nn2) and (l[nearest1]['spec'] != l[nearest2]['spec'])):
        return (['no_confident_result', 'NA', 'no_confident_result'])
    else:
        parent = get_parent(tree, l[query]['node'])
        if parent is None:
            return (['no_confident_result', 'NA', 'no_confident_result'])
        sisterdist = {}
        for sister in get_child_leaves(tree, parent):
            sisterdist[sister.name] = {}
            sisterdist[sister.name]['dist'] = tree.distance(l[query]['node'], sister)
            sisterdist[sister.name]['node'] = sister
            ordered_sisterdist = sorted(sisterdist, key=sisterdist.get)
            for name in ordered_sisterdist:
                if(name != l[query]['id']):
                    forced = re.split("_+", name)
                    return (['no_confident_result', 'NA', forced[-1]])
            return (['no_confident_result', 'NA', 'no_confident_result']) 


def checkclade(query: int, lo: int, hi: int, wc: str, tree: Phylo.BaseTree, l: Dict[int, Dict[str, Any]])-> [str, str]:
    """ recursive substrate prediction for a query & it's sisters in a tree
    Arguments:
        query: index for the query
        lo: index for the lower sister
        hi: index for the higher sister
        wc: wildcard variable
        tree: tree
        l: dictionary of leaf index to str to any
            includes group (str), id (str), spec (str), node (Phylo.node)

    Returns:
        substrate specificity (str), hit name (str)
    """
    if((lo in l) and (hi in l)): ## Not first or last
        if(l[lo]['spec'] == wc): ## lower bound is wildcard
            checkclade(query, lo-1, hi, wc, l)
        elif(l[hi]['spec'] == wc): ## upper bound is wildcard
            checkclade(query, lo, hi+1, wc, l)
        else:
            ## Get the lca's descendants and check specs
            lca = tree.common_ancestor(l[lo]['node'], l[hi]['node'])
            spec = ''
            iname = ''
            passflag = 1
            for child in get_child_leaves(tree, lca):
                split_id = re.split("_+", child.name)
                if(spec != ''): ## assigned
                    if((split_id[-1] != spec) and (split_id[-1] != wc)): ## Specs are different, Requires a deeper dive
                        passflag = 0
                    else:
                        spec = split_id[-1]
                        iname = split_id[0]
                else: ## not yet assigned
                    spec = split_id[-1]
                    iname = split_id[0]
            if(passflag == 0 or spec==''):
                return(['deeperdive', 'NA'])
            else:
                return([spec, iname])
    else: ## First or last
        return(['deeperdive', 'NA'])        


def predicat(tree: Phylo.BaseTree, masscutoff: float, wild: str, snn_thresh: float)-> PredicatResults:
    """ predicat substrate prediction
    Arguments:
        tree: pplacer tree
        masscutoff: cutoff value for pplacer masses
        wild: wildcard variable
        snn_thresh: SNN threshold for confident prediction (default=0.2)

    Returns:
        PredicatResults
            monophyly -> substrate specificity (str)
            forced -> substrate specificity (str)
            snn_score -> scaled nearest neighbor score (float)
    """
    ## predicat settings
    zero_thresh = 0.005 ## Branch distance less than which to call sequences identical
    nppref = ['Q70AZ7_A3', 'Q8KLL5_A3'] ## Leaves used to normalize the nn score to SNN
    npnode = ['', ''] ## initialize node list for leaves above
    dcut = 2.5 ## normalized distance cutoff for nearest neighbor (emperically derived default=2.5)
    query = []
    leaves = {}
    ## Loop through leaves, only keep top placement
    for leaf in tree.get_terminals():
        split_id = re.split("_+", leaf.name) ## Split ID on _, split_id[-1] will be specificity
        if re.match(r"^#[123456789]\d*$", split_id[-2]) is not None: ## matches a non-top pplacer placement
            tree.prune(leaf) ## remove it
    ## Loop through leaves to find groups
    last = '' ## Keep track of the last specificity, initialize on ''
    group = 1 ## group number
    leafnum = 1 ## leaf number
    for leaf in tree.get_terminals():
        if(bool(re.search('^'+nppref[0], leaf.name))): ## if node is nppref[0], store the node
            npnode[0] = leaf
        elif(bool(re.search('^'+nppref[1], leaf.name))): ## if node is nppref[1], store the node
            npnode[1] = leaf
        split_id = re.split("_+", leaf.name) ## Split ID on _, split_id[-1] will be specificity
        if(last != ''): ## Every pass except the first
            if((last != split_id[-1]) or (last != wild)): ## begin new group
                group += 1
        if re.match("^#0$", split_id[-2]) is not None: ## matches pplacer query formatting; #0 is the top placement
            last = wild
            mass = float(re.sub(r"^.+#\d+_M=(\d+?\.?\d*)$", "\g<1>", leaf.name))
            if(mass < masscutoff):
                return PredicatResults('no_confident_result', 'no_confident_result', 0)
        else:
            last = split_id[-1]
        leaves[leafnum] = {}
        leaves[leafnum]['id'] = leaf.name
        leaves[leafnum]['group'] = group
        leaves[leafnum]['spec'] = last
        leaves[leafnum]['node'] = leaf
        ## Record queries
        if(last == wild):
            query.append(leafnum)
        leafnum += 1 
    qnum = next(iter(query))
    ## Get distances to knowns
    distfromquery = {}
    for leafnum in leaves:
        if((qnum != leafnum) and (leaves[leafnum]['spec'] != wild)):
            distfromquery[leafnum] = tree.distance(leaves[qnum]['node'], leaves[leafnum]['node'])
    # Sort distances
    ordered_dist = sorted(distfromquery, key=distfromquery.get)
    ## Get zero distances
    zero_dist = []
    for leafnum in ordered_dist:
        if(distfromquery[leafnum] <= zero_thresh):
            if(distfromquery[leafnum] >= 0):
                zero_dist.append(leafnum)
            else:
                break
    forcedpred = 'no_force_needed'
    pred = 'no_call'
    hit = 'NA'
    if(len(zero_dist) > 0): ## query has zero length known neighbors
        pred = leaves[zero_dist[0]]['spec']
        hit = re.search("^(\S+)_.+$", leaves[zero_dist[0]]['id']).groups()[0]
    else:
        ## predict the clade
        pred, hit = checkclade(qnum, qnum-1, qnum+1, wild, tree, leaves)
        if(pred == 'deeperdive'):
            pred, hit, forcedpred = deeperdive(qnum, tree, ordered_dist[0], ordered_dist[1], leaves)
            if(hit != 'NA'):
                hit = re.search("^(\S+)_.+$", hit).groups()[0]
    if forcedpred == 'no_force_needed':
        forcedpred = pred
    normdist = tree.distance(npnode[0], npnode[1])
    nn_dist = float(distfromquery[ordered_dist[0]]) / normdist
    nnscore = 0
    snnscore = 0
    if(nn_dist < dcut):
        snnscore = getscore(dcut, normdist, distfromquery, leaves, ordered_dist)
        nnscore = calcscore(dcut, nn_dist)
    if(snnscore < snn_thresh):
        forcedpred = 'no_confident_result'
    return PredicatResults(pred, forcedpred, snnscore)


def run_predicat(reference_aln: str, queryfa: Dict[str, str], wildcard: str, ref_tree: str, ref_pkg: str, masscutoff: float, snn_thresh: float) -> PredicatResults:
    """ pplacer and predicat substrate prediciton
    Arguments:
        reference_aln: filename for reference protein fasta, see sandpuma_multithreaded comments for requirements
        queryfa: seq id to seq dictionary
        wildcard: suffix str identifying query sequence (Default= 'UNK' which means headers end in '_UNK')
        ref_tree: reference tree (newick)
        ref_pkg: pplacer reference package
        masscutoff: cutoff value for pplacer masses
        snn_thresh: SNN threshold for confident prediction (default=0.2)

    Returns:                                                                                                                            PredicatResults
            monophyly -> substrate specificity (str)
            forced -> substrate specificity (str)
            snn_score -> scaled nearest neighbor score (float)
    """
    query = next(iter(queryfa))
    ## Align query to a single known sequence
    to_align = {}
    to_align[query] = queryfa[query]
    ref = fasta.read_fasta(reference_aln)
    tname = next(iter(ref)) ## Grab any training sequence header
    to_align[tname] = ref[tname].replace('-', '')
    aligned = subprocessing.run_mafft_predicat_trim(to_align)
    ## trim overhangs
    head = len(re.sub(r'^(-*).+$', r'\g<1>', aligned[tname]))
    tail = len(re.sub(r'^.+(-*)$', r'\g<1>', aligned[tname]))
    trimmed = aligned[query][head:len(aligned[query])-tail].replace('-', '') ## Removes head and tail then removes gaps
    trimmedfa = {query: trimmed}
    ## Align trimmed seq to reference
    all_aligned = subprocessing.run_muscle_profile_sandpuma(reference_aln, trimmedfa)
    ## Pplacer (NOTE: this is new to SANDPUMA as of antiSMASH5 and needs to be tested
    pplacer_tree = subprocessing.run_pplacer(ref_tree, reference_aln, ref_pkg, all_aligned)
    ## prediCAT
    return predicat(pplacer_tree, masscutoff, wildcard, snn_thresh)

def run_asm(queryfa: Dict[str, str], stachfa: Dict[str, str], seedfa: Dict[str, str]) -> [str, int]:
    """ Active site motif (ASM) substrate prediction
    Arguments:
        queryfa: seq id to seq dictionary
        stachfa: seq name to seq for stachelhaus codes
        seedfa: seq name to seq for seed alignment for stachelhaus code extraction

    Returns:                                                                                                                            substrate specificity prediction (str)
        number of identical matches (int)
    """ 
    ## ASM settings
    gapopen = 3.4
    properlen = 117 ## true length
    grsAcode = {4:1,5:1,8:1,47:1,68:1,70:1,91:1,99:1,100:1} ## positions in grsA for code
    ## Alignment
    toalign = {**queryfa, **seedfa}
    aligned2seed = subprocessing.mafft_sandpuma_asm(toalign, gapopen)
    ## Loop through alignment to find new positions for code
    qname = next(iter(queryfa))
    pos = 0
    newcode = []
    for p, val in enumerate(aligned2seed['phe_grsA']):
        if(val=='-'):
            continue
        else:
            pos += 1
            if(pos in grsAcode):
                newcode.append(p)
    ## Extract codes
    extractedcode = {}
    for seqname in aligned2seed:
        code = ''
        for n in newcode:
            code = code + aligned2seed[seqname][n]
            extractedcode[seqname] = code
    ## Error checking
    truth = {'phe_grsA':'DAWTIAAIC', 'asp_stfA-B2':'DLTKVGHIG','orn_grsB3':'DVGEIGSID','val_cssA9':'DAWMFAAVL'}
    for seqname in extractedcode:
        if seqname == qname:
            continue
        else:
            if extractedcode[seqname] != truth[seqname]:
                #print("\t".join([seqname, extractedcode[seqname], truth[seqname]]))
                return('no_call','0') ## Issue with the alignment
    ## Score each
    scores = {}
    for sname in stachfa:
        match = 0
        split_id = re.split("_+", sname)
        if re.match(r"\|", split_id[-1]) is not None: 
            spec = re.split("|", split_id[-1])
        else:
            spec = [split_id[-1]]
        for p, val in enumerate(stachfa[sname]):
            if val == extractedcode[qname][p]:
                match += 1
        if str(match) in scores:
            for s in spec:
                if s in scores[str(match)]:
                    scores[str(match)][s] += 1
                else:
                    scores[str(match)][s] = 1
        else:
            scores[str(match)] = {}
            for s in spec:
                scores[str(match)][s] = 1
    ## Dereplicate and return spec predictions
    for i in range(0,10):
        m = str(9-i)
        if m in scores:
            seen = {}
            for s in scores[m]:
                if s.count('|') > 0:
                    for ss in s.split('|'):
                        seen[ss] = 1
                else:
                    seen[s] = 1
            return('|'.join(sorted(seen)), m )
    return('no_call','0')

def seq2features(seq34: str, phy34: str, s_str: str, p_str: str, predicat_result: PredicatResults, asm_str: str, asm_matches: int, pid: float, i2s: Dict[int, List]) -> [List, List]:
    """ Turns input into feature set for neural network classifier
    Arguments:
        seq34: AA sequence code
        phy34: physicochemical code
        s_str: ordered string for all sequences
        p_str: ordered string for all physicochem
        predicat_result: results from predicat
        asm_str: prediction from ASM
        asm_matches: number of identical residues in ASM
        pid: percent identity to training set
        i2s: map of integers to specificities

    Returns:                                                                                                                            feature list
    """ 
    ## Add PID feature
    features = [ float(pid/100) ] ## Add PID to training set
    ## Add prediCAT features
    for i in range(0, len(i2s)): ## Add prediCAT monophyly result
        if predicat_result.monophyly == i2s[i]:
            features.append(1)
        else:
            features.append(0)
    for i in range(0, len(i2s)): ## Add prediCAT forced result
        if predicat_result.forced == i2s[i]:
            features.append(1)
        else:
            features.append(0)
    features.append( predicat_result.snn_score ) ## Add snn score
    ## Add ASM features
    features.append( float(asm_matches)/9 ) ## Add fraction of ASM code matched
    asm_weight = 1/(1+float(asm_str.count('|')))
    if asm_weight == 1: ## Add ASM matches, if single prediction
        for i in range(0, len(i2s)): 
            if asm_str == i2s[i]:
                features.append(asm_weight)
            else:
                features.append(0)
    else: ## Add ASM matches, if multiple equal predicitons
        for i in range(0, len(i2s)):
            match = 0
            for a in asm_str.split('|'):
                if a == i2s[i]:
                    match = 1
            if match == 1:
                features.append(asm_weight)
            else:
                features.append(0)
    ## Add Seq/Phys features
    for pos in range(0,34):
        for aa in s_str: ## Sequence features
            if seq34[pos] == aa:
                features.append(1)
            else:
                features.append(0)
        for aa in p_str: ## Phys features
            if phy34[pos] == aa:
                features.append(1)
            else:
                features.append(0)
    return features

def extract_seq_features(queryfa: [str, str], ref: str, s_str: str, p_str: str, predicat_result: PredicatResults, asm_str: str, asm_matches: int, pid: float, i2s: Dict[int, str]) -> [List, List]:
    """ Extract feature codes for Neural network (NN) substrate prediction
    Arguments:
        queryfa: seq id to seq dictionary
        ref: reference A domains for alignment
        s_str: seq abbrev string
        p_str: physicochem abbrev string
        predicat_result: results from predicat
        asm_str: prediction from ASM
        asm_matches: number of identical residues in ASM
        pid: percent identity to training set
        i2s: map of integers to specificities
    Returns:                                  
        features
    """
    ## Set positions
    startpos = 66
    a34positions = [210, 213, 214, 230, 234,
                    235, 236, 237, 238, 239,
                    240, 243, 278, 279, 299,
                    300, 301, 302, 303, 320,
                    321, 322, 323, 324, 325,
                    326, 327, 328, 329, 330,
                    331, 332, 333, 334]
    positions34 = []
    for p in a34positions:
        positions34.append(p-startpos)
    aligned = subprocessing.run_muscle_profile_sandpuma(ref, queryfa)
    refname = "P0C062_A1"
    ## Get the 34 code for the query
    qname = next(iter(queryfa))
    refseq = aligned[refname]
    allp, nongaps = 0, 0
    poslist = []
    while refseq != '':
        if nongaps in positions34 and refseq[0] != '-':
            poslist.append(allp)
        if refseq[0] != '-':
            nongaps += 1
        allp += 1
        refseq = refseq[1:]
    seq34 = ''
    for j in poslist:
        aa = aligned[qname][j]
        k, l = j, j
        if aa == '-':
            k += 1
            l = l - 1
            if l not in poslist:
                aa = aligned[qname][l]
            elif (j+1) not in poslist:
                aa = aligned[qname][k]
        seq34 = seq34+aa
    ## E= Positive; N= Negative; U= PolarUncharged; H= Hydrophobic; G= Glycine; C= Cysteine; P= Proline
    pmap = {'R': 'E', 'H': 'E', 'K': 'E',
            'D': 'N', 'E': 'N',
            'S': 'U', 'T': 'U', 'N': 'U', 'Q': 'U',
            'A': 'H', 'V': 'H', 'I': 'H', 'L': 'H', 'M': 'H', 'F': 'H', 'Y': 'H', 'W': 'H',
            'G': 'G', 'C': 'C', 'P': 'P',
            '-': '-'}
    phy34 = ''
    for aa in seq34:
        phy34 = phy34+pmap[aa]
    #print("\t".join([qname, seq34, phy34])) ## Uncomment for pre-computing
    ## Make feature sets
    return(seq2features(seq34, phy34, s_str, p_str, predicat_result, asm_str, asm_matches, pid, i2s))


def run_neuralnet(query_fasta: Dict[str, str], extract_ref: str, nn_clf: MLPClassifier, scaler: StandardScaler, i2s: Dict[int, str], s_str: str, p_str:str, predicat_result: PredicatResults, asm_str: str, asm_matches: int, pid: float) -> NNResults:
    """ Fires the neural network classifier
    Arguments:
        query_fasta: seq id to seq dictionary
        extract_ref: reference A domains for alignment
        nn_clf: trained neural network
        i2s: map of integers to specificities
        s_str: seq abbrev string
        p_str: physicochem abbrev string
        predicat_result: results from predicat
        asm_str: prediction from ASM
        asm_matches: number of identical residues in ASM
        pid: percent identity to training set

    Returns:                                  
        NNResults
    """
    q_features = extract_seq_features(query_fasta, extract_ref, s_str, p_str, predicat_result, asm_str, asm_matches, pid, i2s)
    q_features = np.array(q_features).astype(np.float).reshape(1, -1)
    q_features = scaler.transform(q_features)
    
    all_prob = nn_clf.predict_proba(q_features)

    pred1, pred2, pred3 = '','',''
    prob1, prob2, prob3 = 0,0,0

    i = 0
    for n in all_prob[0]:
        if n > prob1:
            pred3 = pred2
            prob3 = prob2
            pred2 = pred1
            prob2 = prob1
            pred1 = i2s[i]
            prob1 = n
        elif n > prob2:
            pred3 = pred2
            prob3 = prob2
            pred2 = i2s[i]
            prob2 = n
        elif n > prob3:
            pred3 = i2s[i]
            prob3 = n
        i += 1
        
    return( NNResults(pred1, prob1, pred2, prob2, pred3, prob3 ) )
    

def sandpuma_multithreaded(group: str, fasta: Dict[str, str], knownfaa: Dict[str, str], wildcard: str, snn_thresh: float, knownasm: str, ref_aln: str, ref_tree: str, ref_pkg: str, masscutoff: float, stachfa: Dict[str, str], seedfa: Dict[str, str], i2s: List[str], piddb: str, nn_clf: MLPClassifier, scaler: StandardScaler, extract_ref: str, s_str: str, p_str: str) -> Dict[str, SandpumaResults]:
    """ SANDPUMA
    Order of processing:
        predicat: both monophyly and SNN substrate specificity prediction
        asm: active-site motif substrate specificity prediction
        pid: calculate protein percent identity to the known/training set
        neural network: substrate specificity prediction based on the results above and the sequence around the active site

    Arguments:
        group: prefix group name
        fasta: dictionary of seq names (str) to seqs (str)
        knownfaa: dictionary for reference protein fasta; assumes header ends in '_' followed by the <substrate specificity>
        wildcard: str to append to the end of each query sequence; should be different that all specificities (Default= 'UNK')
        snn_thresh: threshold for SNN score (Default= 0.2)
        knownasm: filename for reference active site motif protein fasta, similar header formatting as knownfaa
        ref_aln: reference alignment (fasta) file
        ref_tree: reference tree (newick)
        ref_pkg: pplacer reference package
        masscutoff: cutoff value for pplacer masses
        stachfa: seq name to seq for stachelhaus codes
        seedfa: seq name to seq for seed alignment for stachelhaus code extraction
        i2s: ordered list of specificities
        piddb: diamond db for PID
        nn_clf: trained neural network
        scaler: trained feature scaler
        extract_ref: reference for code extraction alignment
        s_str: ordered sequence string
        p_str: ordered physicochem string

    Returns:                                                                                                                
        dictionary of SandpumaResults
    """
    sp_results = {}
    for query in fasta:
        wc_name = query+'_'+wildcard
        ## Store as a dictionary for functions that don't
        queryfa = {wc_name: fasta[query]}
        ## PrediCAT
        #print("prediCAT")
        predicat_result = run_predicat(ref_aln, queryfa, wildcard, ref_tree, ref_pkg, masscutoff, snn_thresh)
        ## ASM
        #print("ASM")
        asm, asm_matches = run_asm(queryfa, stachfa, seedfa)
        ## PID
        #print("PID")
        pid = subprocessing.run_pid_sandpuma(queryfa, piddb)
        ## NeuralNet
        #print("NeuralNet")
        nn_result = run_neuralnet(queryfa, extract_ref, nn_clf, scaler, i2s, s_str, p_str, predicat_result, asm, asm_matches, pid)
        sp_results[query] = SandpumaResults( predicat_result, asm, pid, nn_result )
    return sp_results


def split_into_groups(fasta: Dict[str, str], n_groups: int) -> Dict[str, List[str]]:
    """ divides a query set into groups to run over SANDPUMA's parallelized pipleline
    Arguments:
        fasta: dictionary of seq names (str) to seqs (str)
        n_groups: number of groups to split into (you can think of this as the number of threads)

    Returns:
        dictionary of groups to fasta headers to seqs
    """
    n_seqs = len(fasta)
    seqs_per_group = int(n_seqs / n_groups)
    qnum = 0
    groupnum = 1
    groups = {}
    for qname in fasta:
        if (qnum == 0) or (qnum < seqs_per_group):
            groupname = 'group'+str(groupnum)
            if groupname not in groups:
                groups[groupname] = {}
            groups[groupname][qname] = fasta[qname]
            qnum += 1
        else:
            groupnum += 1
            groupname = 'group'+str(groupnum)
            groups[groupname] = {}
            groups[groupname][qname] = fasta[qname]
            qnum = 1
    return groups


def run_sandpuma(name2seq: Dict[str, str], threads: int, knownfaa: str, wildcard: str, snn_thresh: float, knownasm: str, jackknife_data: str, ref_aln: str, ref_tree: str, ref_pkg: str, masscutoff:float, seed_file: str, piddb: str, a: float, extract_fa: str, nncodes: str) -> Dict[str, SandpumaResults]:
    """ SANDPUMA parallelized pipleline
    Arguments:
        name2seq: dictionary of seq names (str) to seqs (str)
        threads: number of threads
        knownfaa: filename for reference protein fasta; assumes each header ends in '_' followed by the <substrate specificity>
        wildcard: str to append to the end of each query sequence; should be different that all specificities (Default= 'UNK')
        snn_thresh: threshold for SNN score (Default= 0.2)
        knownasm: filename for reference active site motif protein fasta, similar header formatting as knownfaa
        jackknife_data: filename for jackknife benchmarking results
        ref_aln: reference alignment (fasta) file
        ref_tree: reference tree (newick)
        ref_pkg: pplacer reference package
        masscutoff: cutoff value for pplacer masses
        seed_file: seed fasta file (single entry) used for stachelhaus code extraction
        piddb: diamand db for PID
        a: alpha for neural network (default = 0.1)
        extract_fa: fasta path for nn code extraction
        nncodes: precomputed NN codes for training data

    Returns:                                     
        Dictionary of sandpuma results
    """
    ## Set NN strings
    p_str = 'ENUHGCP-'
    s_str = 'RHKDESTNQAVILMFYWGCP-'
    ## Get all the specificities and map to integers
    allspec = {}
    for name in knownfaa:
        allspec[name.split("_")[-1]] = -1
    i2s = []
    i = 0
    for spec in sorted(allspec, key=allspec.get):
        allspec[spec] = i
        i2s.append(spec)
        i += 1
    allspec['not_in_training'] = i
    i2s.append('not_in_training')
    nn34 = {}
    with open(nncodes, "r") as n:
        for line in n:
            line = line.strip()
            l = line.split("\t")
            nn34[l[0]] = {'seq': l[1],
                          'phy': l[2]}
            spec = l[0].split("_")[-1]
            if spec not in allspec:
                i += 1
                allspec[spec] = i
                i2s.append(spec)
    ## Load jackknife data
    jk = {}
    with open(jackknife_data, "r") as j:
        next(j) ## skip header
        for line in j:
            line = line.strip()
            l = line.split("\t")
            uname = '_'.join([l[0], l[1], l[2]])
            jk[uname] = {'shuf': l[0],
                         'jk': l[1],
                         'query': l[2],
                         'true': l[3],
                         'pid': float(l[4]),
                         'predicat': l[5],
                         'forced_predicat': l[6],
                         'snn': float(l[7]),
                         'asm': l[8],
                         'asm_matches': int(l[9])
                         }
                
    ## Get feature and label sets
    nn_features = []
    nn_labels = []
    #for name in knownfaa:
    for uname in jk:
        name = jk[uname]['query']
        pcr = PredicatResults( jk[uname]['predicat'], jk[uname]['forced_predicat'], jk[uname]['snn'] )
        feature_matrix = seq2features(nn34[name]['seq'],
                                      nn34[name]['phy'],
                                      s_str,
                                      p_str,
                                      pcr,
                                      jk[uname]['asm'],
                                      jk[uname]['asm_matches'],
                                      jk[uname]['pid'], i2s)
        nn_features.append(feature_matrix)
        spec_int = allspec[name.split("_")[-1]]
        nn_labels.append(spec_int)
    nn_features = np.array(nn_features).astype(np.float)
    ## Feature scaling
    scaler = StandardScaler()
    scaler.fit(nn_features)
    nn_features = scaler.transform(nn_features)
    ## Train the neural networks
    nn_clf = MLPClassifier(alpha=a, random_state=1)
    nn_clf = nn_clf.fit(nn_features, nn_labels)
  
    ## Load ASM fastas        
    stach_fa = fasta.read_fasta(knownasm)
    seed_fa = fasta.read_fasta(seed_file)
    ## Split groups
    groups = split_into_groups(name2seq, threads)

    args = []
    for group in groups:
         args.append([group, groups[group], knownfaa, wildcard, snn_thresh, knownasm, ref_aln, ref_tree, ref_pkg, masscutoff, stach_fa, seed_fa, i2s, piddb, nn_clf, scaler, extract_fa, s_str, p_str])
    return(subprocessing.parallel_function(sandpuma_multithreaded, args, cpus=threads))

def xval_sandpuma(knownfaa: str, jackknife_data: str, nncodes: str):
    """ SANDPUMA parallelized pipleline
    Arguments:
        knownfaa: filename for reference protein fasta; assumes each header ends in '_' followed by the <substrate specificity>
        jackknife_data: filename for jackknife benchmarking results
        nncodes: preprocessed AA codes

    Returns:                                     

    """
    ## Set NN strings
    p_str = 'ENUHGCP-'
    s_str = 'RHKDESTNQAVILMFYWGCP-'
    ## Get all the specificities and map to integers
    allspec = {}
    for name in knownfaa:
        allspec[name.split("_")[-1]] = -1
    i2s = []
    all_labels = []
    i = 0
    for spec in sorted(allspec, key=allspec.get):
        allspec[spec] = i
        all_labels.append(i)
        i2s.append(spec)
        i += 1
    allspec['not_in_training'] = i
    i2s.append('not_in_training')
    nn34 = {}
    with open(nncodes, "r") as n:
        for line in n:
            line = line.strip()
            l = line.split("\t")
            nn34[l[0]] = {'seq': l[1],
                          'phy': l[2]}
            spec = l[0].split("_")[-1]
            if spec not in allspec:
                i += 1
                allspec[spec] = i
                i2s.append(spec)
    ## Load jackknife data
    jk = {}
    with open(jackknife_data, "r") as j:
        next(j) ## skip header
        for line in j:
            line = line.strip()
            l = line.split("\t")
            uname = '_'.join([l[0], l[1], l[2]])
            jk[uname] = {'shuf': l[0],
                         'jk': l[1],
                         'query': l[2],
                         'true': l[3],
                         'pid': float(l[4]),
                         'predicat': l[5],
                         'forced_predicat': l[6],
                         'snn': float(l[7]),
                         'asm': l[8],
                         'asm_matches': int(l[9])
                         }
                
    ## Get feature and label sets
    nn_features = []
    nn_labels = []
    #for name in knownfaa:
    for uname in jk:
        name = jk[uname]['query']
        pcr = PredicatResults( jk[uname]['predicat'], jk[uname]['forced_predicat'], jk[uname]['snn'] )
        feature_matrix = seq2features(nn34[name]['seq'],
                                      nn34[name]['phy'],
                                      s_str,
                                      p_str,
                                      pcr,
                                      jk[uname]['asm'],
                                      jk[uname]['asm_matches'],
                                      jk[uname]['pid'], i2s)
        nn_features.append(feature_matrix)
        spec_int = allspec[name.split("_")[-1]]
        nn_labels.append(spec_int)
    nn_features = np.array(nn_features).astype(np.float)

    with open('../flat/sp1.specmap.tsv', "w") as f:
        for i in range(0,len(i2s)):
            f.write("\t".join([str(i), i2s[i]])+"\n")

    test_sizes = [0.05, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 0.95, 0.01, 0.001, 0.99, 0.999]
    n_splits = 500
    all_xvals = {}
    if os.path.isdir('xval') is not True:
        os.mkdir('xval')
    for ts in test_sizes:
        all_xvals[ts] = {}
        for n in range(0,n_splits):
            print("\t".join([str(ts), str(n)]))
            X_train, X_test, y_train, y_test = train_test_split(nn_features, nn_labels, test_size = ts)
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            nn_clf = MLPClassifier(alpha=0.1, random_state=1)
            nn_clf.fit(X_train, y_train)
            predictions = nn_clf.predict(X_test)
            cm = confusion_matrix(y_test,predictions,all_labels)
            np.savetxt('xval/confusion'+str(ts)+'_'+str(n)+'.csv', cm, '%s', ',')

    
def sandpuma_predict(shuffle, jackknife, testfaa, trainfaa, trainaln, trainnwk, trainrefpkg):
    ## Set params
    test_fa = fasta.read_fasta(testfaa)
    threads = 6 ## This will need to be set by antiSMASH upstream
    data_dir = '../flat/'
    knownfaa = fasta.read_fasta(trainfaa)
    wildcard = 'UNK'
    snn_thresh = 0.2
    knownasm = data_dir+'sp1.stach.faa'
    extract_fa = data_dir+'sp1.muscle.fasta'
    jackknife_data = data_dir+'sp1.jkdata.tsv'
    ref_aln = trainaln
    ref_tree = trainnwk ## created with: fasttree -log fullset0_smiles.fasttree.log < fullset0_smiles.afa > fullset0_smiles.fasttree.nwk
    ref_pkg = trainrefpkg ## created with: taxit create --aln-fasta fullset0_smiles.afa --tree-stats fullset0_smiles.fasttree.log --tree-file fullset0_smiles.fasttree.nwk -P fullset0_smiles.fasttree.refpkg -l a_domain
    masscutoff = 0.6
    seed_file = data_dir+'seed.afa'
    piddb = data_dir+'fullset0_smiles.dmnd'
    nncodes = data_dir+'fullset0_smiles.nncodes.tsv'
    a = 0.1

    ## Actually test
    results = run_sandpuma(test_fa, threads, knownfaa, wildcard, snn_thresh, knownasm, jackknife_data, ref_aln, ref_tree, ref_pkg, masscutoff, seed_file, piddb, a, extract_fa, nncodes)
    
    


'''
## Uncomment below to do cross validation...
data_dir = '../flat/'
xval_sandpuma( fasta.read_fasta(data_dir+'sp1.adomains.faa'),
               data_dir+'sp1.jkdata.tsv',
               data_dir+'sp1.nncodes.tsv'
)
'''

## Uncomment below to predict from a fasta input
test_fa = fasta.read_fasta(sys.argv[1])
threads = 6 ## This will need to be set by antiSMASH upstream
data_dir = '../flat/'
knownfaa = fasta.read_fasta(data_dir+'sp1.adomains.faa')
wildcard = 'UNK'
snn_thresh = 0.2
knownasm = data_dir+'sp1.stach.faa'
extract_fa = data_dir+'sp1.muscle.fasta'
jackknife_data = data_dir+'sp1.jkdata.tsv'
ref_aln = data_dir+'sp1.adomains.afa'
ref_tree = data_dir+'sp1.fasttree.nwk' ## created with: fasttree -log sp1.fasttree.log < sp1.adomains.afa > sp1.fasttree.nwk
ref_pkg = data_dir+'sp1.refpkg' ## created with: taxit create --aln-fasta sp1.adomains.afa --tree-stats sp1.fasttree.log --tree-file sp1.fasttree.nwk -P sp1.refpkg -l a_domain
masscutoff = 0.6 ## for pplacer
seed_file = data_dir+'seed.afa'
piddb = data_dir+'sp1.adomains.dmnd'
nncodes = data_dir+'sp1.nncodes.tsv'
a = 0.1 ## for neural net
## Run SANDPUMA
results = run_sandpuma(test_fa, threads, knownfaa, wildcard, snn_thresh, knownasm, jackknife_data, ref_aln, ref_tree, ref_pkg, masscutoff, seed_file, piddb, a, extract_fa, nncodes)

for r in results:
    for q in r:
        l = q.split('_')
        print("\t".join([q,
                         l[-1],
                         str(r[q].pid),
                         r[q].predicat.monophyly,
                         r[q].predicat.forced,
                         str(r[q].predicat.snn_score),
                         r[q].neuralnet.first_pred,
                         str(r[q].neuralnet.first_prob),
                         r[q].neuralnet.second_pred,
                         str(r[q].neuralnet.second_prob),
                         r[q].neuralnet.third_pred,
                         str(r[q].neuralnet.third_prob),
        ]))
