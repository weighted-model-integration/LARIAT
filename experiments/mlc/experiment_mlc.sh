# log filename
LOGFILE="log_mlc.txt"

rm $LOGFILE

# input folder contains the datasets
INPUTFOLDER="input/"
# output folder contains the experiment results
OUTPUTFOLDER="output/"

mkdir $OUTPUTFOLDER

# general seed number
SEED=666

# timeout for support learning
INCALTIMEOUT=300

# timeout for model renormalization
RENORMTIMEOUT=1200

NMIN=5
NMAX=10
NBINS=10
METHOD="det_"$NMIN"_"$NMAX"_"$NBINS

python3 run_mlc.py -f $INPUTFOLDER -s $SEED -o $OUTPUTFOLDER --incal-timeout $INCALTIMEOUT --renorm-timeout $RENORMTIMEOUT  det --n-min $NMIN --n-max $NMAX --n-bins $NBINS 2>&1 | tee -a $LOGFILE

MININSTSLICE=50
ALPHA=0.0
PRIORWEIGHT=0.1
LEAF="piecewise"
ROWSPLIT="rdc-kmeans"
METHOD="mspn_"$MININSTSLICE"_"$ALPHA"_"$PRIORWEIGHT"_"$LEAF"_"$ROWSPLIT

python3 run_mlc.py -f $INPUTFOLDER -s $SEED -o $OUTPUTFOLDER --incal-timeout $INCALTIMEOUT --renorm-timeout $RENORMTIMEOUT --global-norm mspn --min-inst-slice $MININSTSLICE --alpha $ALPHA --prior-weight $PRIORWEIGHT --leaf $LEAF --row-split $ROWSPLIT 2>&1 | tee -a $LOGFILE

