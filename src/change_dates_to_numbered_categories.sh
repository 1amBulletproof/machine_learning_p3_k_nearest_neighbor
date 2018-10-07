#!/bin/sh
sed 's/jan/1/g' $1 | \
sed 's/feb/2/g' | \
sed 's/mar/3/g' | \
sed 's/apr/4/g' | \
sed 's/may/5/g' | \
sed 's/jun/6/g' | \
sed 's/jul/7/g' | \
sed 's/aug/8/g' | \
sed 's/sep/9/g' | \
sed 's/oct/10/g' | \
sed 's/nov/11/g' | \
sed 's/dec/12/g' | \
sed 's/mon/1/g' | \
sed 's/tue/2/g' | \
sed 's/wed/3/g' | \
sed 's/thu/4/g' | \
sed 's/fri/5/g' | \
sed 's/sat/6/g' | \
sed 's/sun/7/g' 
