#!/home/vladas/software/anaconda/bin/python
######################################################################################
from pylab import *
from scipy import *
import sys,os,re
import subprocess
import getopt

#################################
import argparse
parser = argparse.ArgumentParser(description="NAME",
        formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=35))
#parser.add_argument("-p", type=float, help="parameter p",default=2.55)
#parser.add_argument("name",help="name of the log file", default="../md0.log")
#args = parser.parse_args()
######################################################################################

############################# Figure Settings ########################################
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('font',size = '8')
#rc('text', usetex=True)
#fig = figure(dpi=300)
#fig.set_size_inches(18.5,10.5)
######################################################################################
#data = loadtxt(**FILE**);
N = re.compile(r'There.are.(?P<num>[0-9]+).repl')


def main(argv):
	inputfile = ''
	outputfile = ''

	try:
		opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
	except getopt.GetoptError:
		print 'test.py -i <inputfile> -o <outputfile>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print './exch.rex.py -i <inputfile> -o <outputfile>'
			sys.exit()
		elif opt in ("-i", "--ifile"):
			inputfile = arg
		elif opt in ("-o", "--ofile"):
			outputfile = arg
	if len(inputfile)==0: inputfile="./md0.log"
	if len(outputfile)==0: outputfile="Repl_ex.png"
	print 'Input file is "', inputfile
	print 'Output file is "', outputfile
	
	N = re.compile(r'There.are.(?P<num>[0-9]+).repl')

	for line in open(inputfile):
		m = N.search(line)
		if m:
			num = int(m.group('num'))
			print "Found "+str(num)+" replicas"

	#build match string
	remat = ""
	for i in range(num):
		remat = remat + str(i) + "(?P<r" + str(i) +">.*)"

	print remat 
	EX = re.compile(remat)
	X = re.compile(r'x')

	# Get Repl Ex stats from log
        os.system("grep 'Repl ex' %s > Repl_ex_raw.dat" % inputfile)

        # get exchange time interval!
        interval = subprocess.check_output("grep 'Replica exchange interval' %s | tail -1| cut -d ':' -f 2" % inputfile, shell=True)
        timestep = 0.002 #subprocess.check_output("grep 'dt * =' %s | head -1| cut -d '=' -f 2" % inputfile, shell=True)
        dt = int(interval) * float(timestep) # time interval for each exchange in ps

	
	# set structures to store variables
        count = dict((i, 0) for i in range(num-1))
	total = 0

	f = open('Repl_ex_data.dat', 'w')

	# write header
        header = "%-10s" % "# time/ps"
        for i in range(num-1):
                header += " %3dx%-3d" % (i, i+1)
	f.write("%s\n" % header)


	for line in open("Repl_ex_raw.dat"):
		doex = EX.search(line)
		if doex:
			total += 1

			# loop over replica pairs (i, i+1)
			for i in range(num-1):
				idr = "r"+str(i)
				if X.search(doex.group(idr)): 
					count[i] += 1
				
                        if (total % 2 == 0):
				time = dt * total       # in ps

                                printout = "%10d" % time
				
				for i in range(num-1):
					accept = 100 * count[i] / (0.5 * total)
					printout += "%8.2f" % accept

				f.write("%s\n" % printout)


	# Report final result to stdout!
	for i in range(num-1):
		final_accept = 100 * count[i] / (0.5 * total)
		print "Exchange %2d-%2d acceptance = %.2f" % (i, i+1, final_accept)

	
	# Load data for the quick plot!
	data = genfromtxt("Repl_ex_data.dat", skip_footer=1);

	for i, c in zip(range(num), linspace(1, 0, num)):
	        if i == 0: continue # skip time column
		plot(data[:,0], data[:,i], label="%s x  %s" % (i-1, i), color = plt.cm.jet(c))
	xlabel("Time / ps")
	ylim((0, 100))
	ylabel("% of Successful Exchanges")
	title("Replica Exchange Rate")
	plt.legend()	
	plt.savefig(outputfile)
	plt.show()

if __name__ == "__main__":
        main(sys.argv[1:])
