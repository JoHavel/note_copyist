set -ueo pipefail

mv downloaded/muscima-exported-symbols/eight-note-down downloaded/muscima-exported-symbols/eighth-note-down || :
mv downloaded/muscima-exported-symbols/eight-note-up downloaded/muscima-exported-symbols/eighth-note-up || :

SEED=$1
python padd_muscima.py downloaded/MUSCIMA$SEED MUSCIMA --seed $SEED
