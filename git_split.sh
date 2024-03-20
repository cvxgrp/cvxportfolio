#!/usr/bin/env bash
# From https://stackoverflow.com/questions/3887736/keep-git-history-when-splitting-a-file

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 original copy"
  exit 0
fi

git mv $1 $2
git commit -n -m "Split history $1 to $2"
REV=`git rev-parse HEAD`
git reset --hard HEAD^
git mv $1 temp
git commit -n -m "Split history $1 to $2"
git merge $REV
git commit -a -n -m "Split history $1 to $2"
git mv temp $1
git commit -n -m "Split history $1 to $2"