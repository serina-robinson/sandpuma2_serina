# License: GNU Affero General Public License v3 or later
# A copy of GNU AGPL v3 should have been included in this software package in LICENSE.txt.

""" A collection of functions and related classes for executing external
    commands from within antismash.
"""

from io import StringIO
import logging
import multiprocessing
import sys
import os
from subprocess import PIPE, Popen, TimeoutExpired
from tempfile import NamedTemporaryFile
from typing import Dict, List
import warnings
import re
from Bio import Phylo

from helperlibs.wrappers.io import TemporaryDirectory

from antismash.config import get_config

from fasta import write_fasta, read_fasta

# Don't display the SearchIO experimental warning, we know this.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from Bio import SearchIO


class RunResult:
    """ A container for simplifying the results of running a command """
    def __init__(self, command, stdout, stderr, return_code, piped_out, piped_err):
        self.command = command
        self.stdout_piped = piped_out
        self.stderr_piped = piped_err
        if piped_out:
            self.stdout = stdout.decode()
        if piped_err:
            self.stderr = stderr.decode()
        self.return_code = return_code

    def __getattribute__(self, attr):
        if attr == 'stdout' and not self.stdout_piped:
            raise ValueError("stdout was redirected to file, unable to access")
        if attr == 'stderr' and not self.stderr_piped:
            raise ValueError("stderr was redirected to file, unable to access")
        return super().__getattribute__(attr)

    def successful(self) -> bool:
        """ Returns True if the command exited with an exit code of zero """
        return not self.return_code

    def get_command_string(self) -> str:
        """ Returns the command that was run to obtain this result """
        return " ".join(self.command)


def execute(commands, stdin=None, stdout=PIPE, stderr=PIPE, timeout=None) -> RunResult:
    """ Executes commands in a system-independent manner via a child process.

        By default, both stderr and stdout will be piped and the outputs
        accessible.

        Arguments:
            commands: a list of arguments to execute
            stdin: None or input to be piped into the child process
            stdout: if a filename is provided, stdout from the child process
                    will be piped to that file
            stderr: if a filename is provided, stderr from the child process
                    will be piped to that file
            timeout: if provided, the child process will be terminated after
                     this many seconds

        Returns:
            a RunResult object containing any piped output
    """

    if stdin is not None:
        stdin_redir = PIPE
        input_bytes = stdin.encode("utf-8")
    else:
        stdin_redir = None
        input_bytes = None

    try:
        proc = Popen(commands, stdin=stdin_redir, stdout=stdout, stderr=stderr)
        out, err = proc.communicate(input=input_bytes, timeout=timeout)
    except TimeoutExpired:
        proc.kill()
        raise RuntimeError("Child process '%s' timed out after %d seconds" % (
                commands, timeout))

    return RunResult(commands, out, err, proc.returncode, stdout == PIPE,
                     stderr == PIPE)


def parallel_function(function, args, cpus=None, timeout=None) -> list:
    """ Runs the given function in parallel on `cpus` cores at a time.
        Uses separate processes so all args are effectively immutable.

        Both function and args must be picklable (i.e. the function can't be
        declared in a local scope)

        e.g. parallel_function(len, [["a"], ["aa"], ["aaa"]]) -> [1, 2, 3]

        Arguments:
            function: the function to run, cannot be a lambda
            args: a list of lists, containing the arguments for each function call
            cpus: the number of processes to start (defaults to Config.cpus)
            timeout: the maximum time to wait in seconds

        Returns:
            A list of return values of the target function.
    """

    if not cpus:
        cpus = get_config().cpus

    # if only 1 core is to be used, don't fork to run it... this ignores timeout
    if cpus == 1:
        # list() to handle generators, * to expand the list of args
        return [function(*argset) for argset in args]

    pool = multiprocessing.Pool(cpus)
    jobs = pool.starmap_async(function, args)

    timeouts = False

    try:
        results = jobs.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        timeouts = True
    finally:
        pool.terminate()
        pool.join()
    if timeouts:
        raise RuntimeError("Timeout in parallel function:", function)
    return results


def child_process(command) -> int:
    """ Called by multiprocessing's map or map_async method, cannot be locally
        defined """
    try:
        result = execute(command)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        return result.return_code
    except KeyboardInterrupt:
        #  Need to raise some runtime error that is not KeyboardInterrupt, because Python has a bug there
        raise RuntimeError("Killed by keyboard interrupt")
    return -1


def verbose_child_process(command) -> int:
    """ A wrapper of child_process() that logs the command being run """
    logging.debug("Calling %s", " ".join(command))
    return child_process(command)


def parallel_execute(commands, cpus=None, timeout=None, verbose=True) -> List[int]:
    """ Limited return vals, only returns return codes
    """
    if verbose:
        runner = verbose_child_process
    else:
        runner = child_process
    os.setpgid(0, 0)
    if not cpus:
        cpus = get_config().cpus
    pool = multiprocessing.Pool(cpus)
    jobs = pool.map_async(runner, commands)

    try:
        errors = jobs.get(timeout=timeout)
    except multiprocessing.TimeoutError:
        pool.terminate()
        raise RuntimeError("One of %d child processes timed out after %d seconds" % (
                cpus, timeout))
    except KeyboardInterrupt:
        logging.error("Interrupted by user")
        pool.terminate()
        raise

    pool.close()

    return errors


def run_hmmsearch(query_hmmfile: str, target_sequence: str, use_tempfile=False):
    """ Run hmmsearch on a HMM file and a fasta input

        Arguments:
            query_hmmfile: the path to the HMM file
            target_sequence: the fasta input to search as a string
            use_tempfile: if True, a tempfile will be written for the fasta input
                          instead of piping

        Returns:
            a list of hmmsearch results as parsed by SearchIO
    """
    config = get_config()
    command = ["hmmsearch", "--cpu", str(config.cpus),
               "-o", os.devnull,  # throw away the verbose output
               "--domtblout", "result.domtab",
               query_hmmfile]

    # Allow for disabling multithreading for HMMer3 calls in the command line
    if config.get('hmmer3') and 'multithreading' in config.hmmer3 and \
            not config.hmmer3.multithreading:
        command = command[0:1] + command[3:]

    with TemporaryDirectory(change=True):
        try:
            if use_tempfile:
                with open("input.fa", 'w') as handle:
                    handle.write(target_sequence)
                command.append("input.fa")
                run_result = execute(command)
            else:
                command.append('-')
                run_result = execute(command, stdin=target_sequence)
        except OSError:
            return []
        if not run_result.successful():
            logging.error('hmmsearch returned %d: %s while searching %s',
                          run_result.return_code, run_result.stderr, query_hmmfile)
            raise RuntimeError("Running hmmsearch failed.")
        return list(SearchIO.parse("result.domtab", 'hmmsearch3-domtab'))


def run_hmmpress(hmmfile: str) -> RunResult:
    "Run hmmpress"
    command = ['hmmpress', hmmfile]
    run_result = execute(command)
    if not run_result.successful():
        logging.error("hmmpress failed for file: %s", hmmfile)
    return run_result


def run_hmmpfam2(query_hmmfile: str, target_sequence: str) -> List:  # TODO cleanup
    """ Run hmmpfam2 over the provided HMM file and fasta input

        Arguments:
            query_hmmfile: the HMM file to use
            target_sequence: a string in fasta format of the sequence to run

        Returns:
            a list of results as parsed by SearchIO
    """
    config = get_config()
    command = ["hmmpfam2", "--cpu", str(config.cpus),
               query_hmmfile, '-']

    # Allow to disable multithreading for HMMer2 calls in the command line #TODO fix options for this
    if config.get('hmmer2') and 'multithreading' in config.hmmer2 and \
            not config.hmmer2.multithreading:
        command = command[0:1] + command[3:]

    result = execute(command, stdin=target_sequence)
    if not result.successful():
        logging.debug('hmmpfam2 returned %d: %r while searching %r', result.return_code,
                      result.stderr, query_hmmfile)
        raise RuntimeError("hmmpfam2 problem while running %s", command)
    res_stream = StringIO(result.stdout)
    results = list(SearchIO.parse(res_stream, 'hmmer2-text'))
    return results


def run_fimo_simple(query_motif_file: str, target_sequence: str) -> RunResult:  # TODO cleanup
    """ Runs FIMO on the provided inputs

        Arguments:
            query_motif_file: the path to the file containing query motifs
            target_sequence: the path to the file containing input sequences

        Returns:
            a RunResult with the execution results
    """
    command = ["fimo", "--text", "--verbosity", "1", query_motif_file, target_sequence]
    result = execute(command)
    if not result.successful():
        logging.debug('FIMO returned %d: %r while searching %r', result.return_code,
                      result.stderr, query_motif_file)
        raise RuntimeError("FIMO problem while running %s... %s", command, result.stderr[-100:])
    return result.stdout


def run_hmmscan(target_hmmfile: str, query_sequence: str, opts=None, results_file=None) -> List:
    """ Runs hmmscan on the inputs and return a list of QueryResults

        Arguments:
            target_hmmfile: the path to a HMM file to use in scanning
            query_sequence: a string containing input sequences in fasta format
            opts: a list of extra arguments to pass to hmmscan, or None
            results_file: a path to keep a copy of hmmscan results in, if provided

        Returns:
            a list of QueryResults as parsed from hmmscan output by SearchIO

    """
    if not query_sequence:
        raise ValueError("Cannot run hmmscan on empty sequence")

    config = get_config()
    command = ["hmmscan", "--cpu", str(config.cpus), "--nobias"]

    # Allow to disable multithreading for HMMer3 calls in the command line
    if config.get('hmmer3') and 'multithreading' in config.hmmer3 and \
            not config.hmmer3.multithreading:  # TODO: ensure working
        command = command[0:1] + command[3:]

    if opts is not None:
        command.extend(opts)
    command.extend([target_hmmfile, '-'])
    result = execute(command, stdin=query_sequence)
    if not result.successful():
        raise RuntimeError('hmmscan returned %d: %r while scanning %r' % (
                           result.return_code, result.stderr[-100:].replace("\n", ""),
                           query_sequence[:100]))
    if results_file is not None:
        with open(results_file, 'w') as fh:
            fh.write(result.stdout)

    return list(SearchIO.parse(StringIO(result.stdout), 'hmmer3-text'))


def run_muscle_single(seq_name: str, seq: str, comparison_file: str) -> Dict[str, str]:
    """ Runs muscle over a single sequence against a comparison file in profile
        mode and returns a dictionary of the resulting alignments

        Arguments:
            seq_name: the name of the query
            seq: the sequence to align
            comparison_file: the path of the file containing comparison sequences

        Returns:
            a dictionary mapping sequence name (query or reference) to alignment
    """
    with NamedTemporaryFile(mode="w+") as temp_in:
        with NamedTemporaryFile(mode="w+") as temp_out:
            write_fasta([seq_name], [seq], temp_in.name)
            # Run muscle and collect sequence positions from file
            result = execute(["muscle", "-profile", "-quiet",
                              "-in1", comparison_file,
                              "-in2", temp_in.name,
                              "-out", temp_out.name])
            if not result.successful():
                raise RuntimeError("muscle returned %d: %r while comparing query named %s" % (
                                   result.return_code, result.stderr.replace("\n", ""),
                                   seq_name))
            fasta = read_fasta(temp_out.name)
    return fasta


def align_domain_to_reference(reference_alignment: str, seq_names: list, seq_seqs: list) -> Dict[str, str]:
    """ Aligns PROTEIN domains to a reference PROTEIN alignment via
         muscle then muscle profile

        Arguments:
            reference_alignment: path to the reference alignment
            seq_names: list of all the sequence names (str)
            seq_seqs: list of protein sequences (str) (NOTE: assumes same index as seq_names)

        Returns:
            a fasta alignment
    """
    config = get_config()

    ## Align all queries to each other
    with NamedTemporaryFile(mode="w+") as query_unaligned:
        write_fasta(seq_names,seq_seqs, query_unaligned.name)
        with NamedTemporaryFile(mode="w+") as query_aligned:
            result = execute(["muscle", "-quiet",
                              "-in", query_unaligned.name,
                              "-out", query_aligned.name])
            if not result.successful():
                raise RuntimeError("muscle returned %d: %r while aligning query %s" % (
                    result.return_code, result.stderr.replace("\n", ""),
                    query_unaligned.name))
            with NamedTemporaryFile(mode="w+") as profile_aligned:
                profile_result = execute(["muscle", "profile", "-quiet",
                                          "-in1", reference_alignment,
                                          "-in2", query_aligned.name,
                                          "-out", profile_aligned.name])
                if not profile_result.successful():
                    raise RuntimeError("muscle returned %d: %r while aligning query %s to ref %s" % (
                        profile_result.return_code, profile_result.stderr.replace("\n", ""),
                        query_aligned.name, reference_alignment))
                afa = read_fasta(profile_aligned.name)
    return afa


def run_pplacer(reference_tree: str, reference_alignment: str, reference_package: str, alignment: Dict[str, str]) -> Phylo.BaseTree:
    """ Runs pplacer, places query sequences onto a precalculated phylogeny

        Arguments:
            reference_alignment: path to the reference alignment
            alignment: fasta Dictionary of seq names (str) to seqs (str)

        Returns:
            path to the resulting pplacer singular tree, newick format
    """
    config = get_config()

    anames, aseqs = [], []
    for k, v in alignment.items():
        anames.append(k)
        aseqs.append(v)
        
    with NamedTemporaryFile(mode="w+", suffix='.fasta') as aln:
        write_fasta(anames, aseqs, aln.name)
        with NamedTemporaryFile(mode="w+", suffix='.jplace') as jplace:
            pplacer_result = execute(["pplacer",
                          "-t", reference_tree,
                          "-r", reference_alignment,
                          "-o", jplace.name,
                          "-c", reference_package,
                          aln.name])
            if not pplacer_result.successful():
                raise RuntimeError("pplacer returned %d: %r while placing query alignment %s" % (
                    pplacer_result.return_code, pplacer_result.stderr.replace("\n", ""),
                    aln.name))
            ## Summarize to a single tree
            with NamedTemporaryFile(mode="w+", suffix='.nwk') as gup:
                guppy_result = execute(["guppy", "sing", jplace.name, "-o", gup.name])
                if not guppy_result.successful():
                    raise RuntimeError("guppy sing returned %d: %r while summarizing %s" % (
                        guppy_result.return_code, guppy_result.stderr.replace("\n", ""),
                        jplace.name))
                pplacer_tree = Phylo.read(gup.name, 'newick')
    return(pplacer_tree)

def run_mafft_predicat_trim(fa: Dict[str, str]) -> Dict[str, str]:
    """ Runs mafft for overhang trimming within predicat (SANDPUMA)

        Arguments:
            fa: fasta Dictionary of seq names (str) to seqs (str)

        Returns:
            aligned fasta Dictionary of seq names (str) to seqs (str)
    """
    fnames, fseqs = [], []
    for k, v in fa.items():
        fnames.append(k)
        fseqs.append(v)
    
    with NamedTemporaryFile(mode="w+") as query_unaligned:
        write_fasta(fnames, fseqs, query_unaligned.name)
        with NamedTemporaryFile(mode="w+") as query_aligned:
            mafft_result = execute(["mafft", "--quiet",
                                    "--namelength", "70",
                                    "--op",  "5",
                                    query_unaligned.name],
                                   stdout=query_aligned) ## Note: stdout requires a file object, not a name
            if not mafft_result.successful():
                raise RuntimeError("mafft returned %d: %r while performing initial prediCAT trim alignment" % (
                    mafft_result.return_code, mafft_result.stderr.replace("\n", "") ))
            return read_fasta(query_aligned.name)

def run_muscle_profile_sandpuma(ref_afa: str, query_fa: Dict[str, str]) -> Dict[str, str]:
    """ Runs muscle profile within predicat or svms (SANDPUMA)
        Arguments:
            ref_afa: aligned fasta
            query_fa: single entry fasta Dictionary of seq names (str) to seqs (str)

        Returns:
            aligned fasta Dictionary of all seq names (str) to seqs (str)
    """
    fnames, fseqs = [], []
    for k, v in query_fa.items():
        fnames.append(k)
        fseqs.append(v)
    with NamedTemporaryFile(mode="w+") as query_unaligned:
        write_fasta(fnames, fseqs, query_unaligned.name)
        with NamedTemporaryFile(mode="w+") as all_aligned:
            muscle_result = execute(["muscle", "-profile",
                                     "-in1", ref_afa,
                                     "-in2", query_unaligned.name,
                                     "-out", all_aligned.name])
            if not muscle_result.successful():
                raise RuntimeError("muscle -profile returned %d: %r while performing prediCAT reference alignment" % (
                    muscle_result.return_code, muscle_result.stderr.replace("\n", "") ))
            return read_fasta(all_aligned.name)


def mafft_sandpuma_asm(toalign: Dict[str, str], gapopen: float) -> Dict[str, str]:
    """ Runs mafft for initial alignment for stachelhaus code extraction (SANDPUMA)

        Arguments:
            toalign: fasta Dictionary of seq names (str) to seqs (str)
            gapopen: gapopen penalty


        Returns:
            aligned fasta Dictionary of all seq names (str) to seqs (str)
    """
    fnames, fseqs = [], []
    for k, v in toalign.items():
        fnames.append(k)
        fseqs.append(v)
    with NamedTemporaryFile(mode="w+") as query_unaligned:
        write_fasta(fnames, fseqs, query_unaligned.name)
        with NamedTemporaryFile(mode="w+") as all_aligned:
            mafft_result = execute(["mafft", "--quiet",
                                     "--op", str(gapopen),
                                     "--namelength", "60",
                                     query_unaligned.name],
                                   stdout=all_aligned)
            if not mafft_result.successful():
                raise RuntimeError("mafft returned %d: %r while performing ASM seed alignment" % (
                    mafft_result.return_code, mafft_result.stderr.replace("\n", "") ))
            return read_fasta(all_aligned.name)

def run_svm_sandpuma(seq: str, nrpsdir: str) -> str:
    """ Runs SVM predictions based on NRPSPredictor2 as part of SANDPUMA

        Arguments:
            seq: 34 char A-domanin code
            nrpsdir: NRPSPredictor2 directory


        Returns:
            SVM prediction
    """
    libs = nrpsdir+'/lib'
    build = nrpsdir+'/build'
    datadir = nrpsdir+'/data'

    os.environ['NRPS2BASEDIR'] = nrpsdir
    os.environ['LIBS'] = libs
    os.environ['BUILD'] = build
    os.environ['DATADIR'] = datadir

    pred = 'no_call'
    
    with NamedTemporaryFile(mode="w+") as sig:
        out_file = open(sig.name, "w")
        out_file.write("%s\t%s\n" % (seq, 'query'))
        out_file.close()
        with NamedTemporaryFile(mode="w+") as rep:
            with NamedTemporaryFile(mode="w+") as o:
                svm_result = execute(["java",
                                      "-Ddatadir="+datadir,
                                      "-cp", ':'.join([build+'/NRPSpredictor2.jar',
                                                       libs+'/java-getopt-1.0.13.jar',
                                                       libs+'/Utilities.jar',
                                                       libs+'/libsvm.jar']),
                                      "org.roettig.NRPSpredictor2.NRPSpredictor2",
                                      "-i", sig.name,
                                      "-r", rep.name,
                                      "-s", "1"],
                                     stdout=o)
                if not svm_result.successful():
                    raise RuntimeError("SVM processing returned %d: %r" % (
                        svm_result.return_code, svm_result.stderr.replace("\n", "") ))
                with open(rep.name, "r") as r:
                    for line in r:
                        line = line.strip()
                        if not line:
                            continue
                        if line[0] == '#':
                            continue
                        result = line.split("\t")
                        pred = result[6]
                        break
    return pred

def run_phmm_sandpuma(queryfa: Dict[str, str], hmmdb: str) -> str:
    """ Runs pHMM predictions as part of SANDPUMA

        Arguments:
            queryfa: single entry fasta Dictionary of seq names (str) to seqs (str)
            hmmdb: path to the hmmdb

        Returns:
            pHMM prediction
    """
    fnames, fseqs = [], []
    for k, v in queryfa.items():
        fnames.append(k)
        fseqs.append(v)
    with NamedTemporaryFile(mode="w+") as query:
        write_fasta(fnames, fseqs, query.name)
        with NamedTemporaryFile(mode="w+") as out:
            with NamedTemporaryFile(mode="w+") as tbl:
                phmm_result = execute(["hmmscan",
                                       "-o", out.name,
                                       "--tblout", tbl.name,
                                       "--noali",
                                       hmmdb,
                                       query.name])
                if not phmm_result.successful():
                    raise RuntimeError("SANDPUMA pHMM processing returned %d: %r" % (
                        phmm_result.return_code, phmm_result.stderr.replace("\n", "") ))
                if os.path.exists(tbl.name):
                    with open(tbl.name, "r") as res:
                        for line in res:
                            line = line.strip()
                            if re.match("^\w+\S*\s", line) is not None:
                                hit = re.sub(r"^(\w+\S*)\s.+$", "\g<1>", line)
                                hit = re.sub(r"^(.+)_\S+$", "\g<1>", hit)
                                return hit
                else:
                    return 'no_call'

def run_pid_sandpuma(queryfa: Dict[str, str], db: str) -> float:
    """ Runs protein percent ID calculation for SANDPUMA

        Arguments:
            queryfa: single entry fasta Dictionary of seq names (str) to seqs (str)
            knownfaa: path to the training fasta

        Returns:
            percent identity
    """
    fnames, fseqs = [], []
    for k, v in queryfa.items():
        fnames.append(k)
        fseqs.append(v)
    with NamedTemporaryFile(mode="w+") as query:
        write_fasta(fnames, fseqs, query.name)
        with NamedTemporaryFile(mode="w+") as bp:
            with NamedTemporaryFile(mode="w+") as o:
                bp_result = execute(["diamond", "blastp",
                                     "--query", query.name,
                                     "--db", db,
                                     "--out", bp.name,
                                     "--evalue", "1e-10",
                                     "--query-cover", "50"],
                                    stdout=o)
                if not bp_result.successful():
                    raise RuntimeError("diamond blastp (SANDPUMA pid) returned %d: %r" % (
                        bp_result.return_code, bp_result.stderr.replace("\n", "") ))
                bit, pid = 0, 0
                with open(bp.name, "r") as res:
                    for line in res:
                        line = line.strip()
                        l = line.split("\t")
                        if bit < float(l[-1]):
                            pid = float(l[2])
                            bit = float(l[-1])
                return pid
