# general seed number
SEED=666

# timeout for support learning
INCALTIMEOUT=300

# timeout for model renormalization
RENORMTIMEOUT=1200

# number of samples used in the approximate measures
NSAMPLES=1000000

# number of instances for each parameter configuration
NPROBLEMS=20

# number of training and validation examples
PTRAIN=500
PVALID=50

# fixed generator parameters
D=2
H=5
R=3

# variable generator parameters
LISTB="1 2 3 4 5"
LISTL="2 3 4"

# log filename
EXPNAME="aaai20-lb"
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

	python3 run_support_learning.py -t $INCALTIMEOUT -e $INPUTFILE -s $SEED 2>&1 | tee -a $LOGFILE

	OUTPUTFILE=$OUTPUTFOLDER$EXPNAME"_"$R"_"$B"_"$D"_"$PTRAIN"_"$PVALID"_"$H"_"$L".volume_results"
	python3 run_volume.py -e $INPUTFILE -o $OUTPUTFILE -n $NSAMPLES -t $INCALTIMEOUT -s $SEED | tee -a $LOGFILE

	NMIN=5
	NMAX=10
	NBINS=10
	METHOD="det_"$NMIN"_"$NMAX"_"$NBINS
	INPUTFILE=$INPUTFOLDER$EXPNAME"_"$R"_"$B"_"$D"_"$PTRAIN"_"$PVALID"_"$H"_"$L
	OUTPUTFILE=$OUTPUTFOLDER$EXPNAME"_"$R"_"$B"_"$D"_"$PTRAIN"_"$PVALID"_"$H"_"$L"_"$METHOD

	python3 run_synthetic.py -e $INPUTFILE -s $SEED -o $OUTPUTFILE -t $RENORMTIMEOUT --n-samples $NSAMPLES det --n-min $NMIN --n-max $NMAX --n-bins $NBINS 2>&1 | tee -a $LOGFILE

	MININSTSLICE=50
	ALPHA=0.0
	PRIORWEIGHT=0.1
	LEAF="piecewise"
	ROWSPLIT="rdc-kmeans"
	METHOD="mspn_"$MININSTSLICE"_"$ALPHA"_"$PRIORWEIGHT"_"$LEAF"_"$ROWSPLIT
	
	OUTPUTFILE=$OUTPUTFOLDER$EXPNAME"_"$R"_"$B"_"$D"_"$PTRAIN"_"$PVALID"_"$H"_"$L"_"$METHOD
	
	python3 run_synthetic.py -e $INPUTFILE -s $SEED -o $OUTPUTFILE -t $RENORMTIMEOUT --n-samples $NSAMPLES --global-norm mspn --min-inst-slice $MININSTSLICE --alpha $ALPHA --prior-weight $PRIORWEIGHT --leaf $LEAF --row-split $ROWSPLIT 2>&1 | tee -a $LOGFILE


    done
done

sh plot_everything.sh $OUTPUTFOLDER
