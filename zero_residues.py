from pymol import cmd, stored
import pymol

def zero_residues(sel1,offset=0,chains=0):
        """
DESCRIPTION

    Renumbers the residues so that the first one is zero, or offset

USAGE

    zero_residues selection [, offset [, chains ]]

EXAMPLES

    zero_residues protName            # first residue is 0
    zero_residues protName, 5         # first residue is 5
    zero_residues protName, chains=1  # each chain starts at 0
    zero_residues *
        """
        offset = int(offset)

        # variable to store the offset
        stored.first = None
        # get the names of the proteins in the selection

        names = ['(model %s and (%s))' % (p, sel1)
                        for p in cmd.get_object_list('(' + sel1 + ')')]

        if int (chains):
                names = ['(%s and chain %s)' % (p, chain)
                                for p in names
                                for chain in cmd.get_chains(p)]

        # for each name shown
        for p in names:
                # get this offset
                ok = cmd.iterate("first %s and polymer and n. CA" % p,"stored.first=resv")
                # don't waste time if we don't have to
                if not ok or stored.first == offset:
                        continue;
                # reassign the residue numbers
                cmd.alter("%s" % p, "resi=str(int(resi)-%s)" % str(int(stored.first)-offset))
                # update pymol

        cmd.rebuild()

#pymol.finish_launching()


#pymol.finish_launching(['pymol', '-cq'])

# let pymol know about the function
cmd.extend("zero_residues", zero_residues)
#cmd.run("./test_cmds.pml")
#cmd.run("./test_pymol_a.pml")

