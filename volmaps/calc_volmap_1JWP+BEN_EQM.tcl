# load molecules
mol new {/home/vladas/Dropbox/IPython_notebooks/JACS_2016/TEM1/ref_1JWP+WAT+BEN.gro} type {gro} first 0 last -1 step 1 waitfor 1
# remove first frame
animate delete  beg 0 end 0 skip 0


## LOOP 1! different trajs
for {set i 0} {$i <= 31} {incr i} {

set trajfile /home/vladas/mnt/IB-server/storage/project_UCB/TEM1/multi_NVT/protein+BEN/fixed_traj/traj${i}_aligned_full.xtc


## LOOP 2! divide traj blocks
for {set x 0} {$x <= 10} {incr x} {

set begin [expr $x * 1000 ]
set end [expr $x * 1000 + 1000 ]
set outfile density-DM_1JWP+BEN_EQM-traj${i}_frames-$begin-$end.dx

# load traj chunk!
mol addfile $trajfile type {xtc} first $begin last $end step 1 waitfor -1
# do density calculation!
volmap density [atomselect top "name DM"] -res 1.0 -radscale 5.0 -allframes -combine avg -o $outfile -minmax {{-45 -60 -25} {65 60 95}} -checkpoint 0
# remove coordinates
animate delete  beg 0 end -1 skip 0

}
}
