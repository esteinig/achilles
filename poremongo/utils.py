import os
import sys
import shlex
import subprocess

def run_cmd(cmd, callback=None, watch=False, shell=False):

    """Runs the given command and gathers the output.

    If a callback is provided, then the output is sent to it, otherwise it
    is just returned.

    Optionally, the output of the command can be "watched" and whenever new
    output is detected, it will be sent to the given `callback`.

    Returns:
        A string containing the output of the command, or None if a `callback`
        was given.
    Raises:
        RuntimeError: When `index` is True, but no callback is given.

    """

    if watch and not callback:
        raise RuntimeError(
            'You must provide a callback when watching a process.'
        )

    output = None
    try:
        if shell:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                shell=True,
                preexec_fn=os.setsid
            )
        else:
            proc = subprocess.Popen(
                shlex.split(cmd),
                stdout=subprocess.PIPE
            )
        if watch:
            while proc.poll() is None:
                line = proc.stdout.readline()
                if line != "":
                    callback(line)

            # Sometimes the process exits before we have all of the output, so
            # we need to gather the remainder of the output.
            remainder = proc.communicate()[0]
            if remainder:
                callback(remainder)
        else:
            output = proc.communicate()[0]
    except:
        err = str(sys.exc_info()[1]) + "\n"
        output = err

    if callback and output is not None:
        return callback(output)

    return output

