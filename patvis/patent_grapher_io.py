import os
from alife.visualize.patent_grapher import PatentGrapher
from alife.visualize import get_family_and_friends

faf = get_family_and_friends()

class PatentGrapherIO(object):

    def __init__(self):
        self.pg = PatentGrapher()
    
    def load_style(self,filename=".graph_style"):
            if os.path.isfile(filename):
                with open(filename,"r") as f:
                    for line in f.readlines():
                        commands = line.strip(' \n')
                        self.read_commands(commands)

    def __str__(self):
        return self.pg.__str__()

    def read_commands(self,resp):

        if resp.strip().startswith("graph"):

            pno, numgens, name, threshold = resp.strip().split()[1:]
            numgens, threshold = int(numgens), int(threshold)

            if pno.isdigit():
                self.pg.build_and_draw_pno_graph(int(pno), numgens, name,
                                                 threshold=threshold)
            elif pno == 'faf':
                self.pg.gen_faf_graphs(pno, numgens, threshold)
            else:
                if threshold == -1:
                    threshold = faf[pno][1]
                self.pg.build_and_draw_pno_graph(faf[pno][0], numgens, name,
                                                 threshold=threshold)

        elif resp.strip().startswith("set"):
            param, value = resp.strip().split()[1:]
            print(param, value)
            if isinstance(self.pg.__dict__[param],str):

                if param == "trait_type":
                    self.pg.set_trait_type(value)
                elif param == "db":

                    try:
                        socket.inet_aton(value[0])
                        value[1] = int(value[1])
                        if not isinstance(value[2],string):
                            raise(ValueError)
                        set_database(*value)
                    except socket.error as ValueError:
                        print("Error: The database connection information should be in the form <ip address> <port number> <collection name>")

                else:
                    self.pg.__dict__[param] = value

            elif isinstance(self.pg.__dict__[param],bool):
                self.pg.__dict__[param] = (value == "True" or value == "true" or value == "T" or value == "t")
            elif isinstance(self.pg.__dict__[param],int):
                self.pg.__dict__[param] = value
            elif isinstance(self.pg.__dict__[param],tuple):
                self.pg.__dict__[param] = tuple(map(int, value.split(',')))

        elif resp.strip().startswith("help"):
            print("""Available commands:
            graph <patent nickname> <number of generations> <output filename> <citation threshold>
            : graphs a familiar patent, see command faf for nicknames

            graph <patent number> <number of generations> <output filename> <citation threshold>
            : graphs any patent

            set <parameter> <value>
            : sets a graphing parameter to the desired value

            load <filename>
            : loads a file of commands and executes them
              - Any file in the current working directory names '.graph_style' will be loaded by default on program start

            printenv
            : displays your current graphing environment

            faf
            : prints patent nicknames for family and friends

            help
            : displays available commands

            q
            : exits the program

            Example graph command:
                        graph minesweeper 3 minesweeper.pdf 1

            The output filename will have extra qualifiers to differentiate it from other graphs of the same patent.

            Available traits:
            category
            subcategory
            class
            w2v
            """)

        elif resp.strip().startswith("printenv"):
            print(self)

        elif resp.strip() == "faf":
            print("\nFamily and Friends:\n\t" + '\n\t'.join(str(patn) for patn in sorted(faf.keys())) + "\n")

        elif resp.strip().startswith("load"):
            filename = resp.strip().split()[1]
            self.load_style(filename=filename)

        elif resp.strip() == "q":
            if self.pg.conn:
                self.pg.conn.close()
            return True

        else:
            print("Malformed Command: Type \'help\' for options")

def main():
    pgio = PatentGrapherIO()
    pgio.load_style()
    print(pgio)
    
    Break = False
    while(not Break):
        resp = input("Please enter a command:\n")
        Break = pgio.read_commands(resp)

if __name__ == '__main__':
    main()
