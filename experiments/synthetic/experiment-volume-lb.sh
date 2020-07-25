
# general seed number
SEED=666

# timeout for INCAL+
INCALTIMEOUT=600

# number of negative samples for bootstrapping INCAL+
NEGATIVEBOOTSTRAP=50

# number of samples used in the approximate measures
NSAMPLES=1000000

# number of instances for each parameter configuration
NPROBLEMS=10

# number of training and validation examples
PTRAIN=100
PVALID=50

# fixed generator parameters
D=2
H=5
R=3

# variable generator parameters
LISTB="2 3 4 5 6"
LISTL="2 3 4"

# log filename
EXPNAME="volume-lb2-"$PTRAIN"-"$PVALID
LOGFILE="log_"$EXPNAME".log"

rm $LOGFILE

# input folder contains the datasets, ground truth models, learned supports
INPUTFOLDER="input-"$EXPNAME"/"
# output folder contains the learned models as well as evaluation results
OUTPUTFOLDER="output-"$EXPNAME"/"

mkdir $INPUTFOLDER
mkdir $OUTPUTFOLDER

for B in $LISTB
do
    for L in $LISTL
    do

	OUTPUTFILE=$INPUTFOLDER$EXPNAME"_"$R"_"$B"_"$D"_"$PTRAIN"_"$PVALID"_"$H"_"$L

	python3 generate_experiment.py -o $OUTPUTFILE -s $SEED -n $NPROBLEMS -t $PTRAIN -v $PVALID -r $R -b $B -d $D --hyperplanes $H --literals $L 2>&1 | tee -a $LOGFILE

	INPUTFILE=$OUTPUTFILE
	OUTPUTFILE=$OUTPUTFOLDER$EXPNAME"_"$R"_"$B"_"$D"_"$PTRAIN"_"$PVALID"_"$H"_"$L"_volume"
	python3 run_volume.py -t $INCALTIMEOUT -e $INPUTFILE -s $SEED --negative-bootstrap $NEGATIVEBOOTSTRAP --n-samples $NSAMPLES -o $OUTPUTFILE 2>&1 | tee -a $LOGFILE

    done
done

