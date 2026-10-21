#!/bin/bash
# Чтение ВСЕЙ английской Википедии по томам (multistream, ~60 файлов). Каждый том: скачать → прочитать → удалить bz2.
# Память: enwiki_store_NN. Ключи ~650 МБ/том. Перезапускаемый: пропускает тома, у которых есть store/facts.jsonl.
set -u
cd ~/Developer/mini-ai-memory/wt
DATA=~/Developer/mini-ai-data; OUT=~/Developer/mini-ai-runs/memory-core
LIST=$DATA/enwiki_parts.txt
if [ ! -s "$LIST" ]; then
  curl -s https://dumps.wikimedia.org/enwiki/latest/ | grep -oE 'enwiki-latest-pages-articles-multistream[0-9]+\.xml-p[0-9]+p[0-9]+\.bz2' | sort -u > "$LIST"
fi
N=$(wc -l < "$LIST"); echo "томов: $N"
i=0
while read -r part; do
  i=$((i+1)); tag=$(printf "%02d" $i); store=$OUT/enwiki_store_$tag
  if [ -s "$store/facts.jsonl" ]; then echo "[$tag] есть, пропуск"; continue; fi
  echo "[$tag] $part"; t0=$(date +%s)
  curl -sL -o "$DATA/cur.xml.bz2" "https://dumps.wikimedia.org/enwiki/latest/$part" || { echo "[$tag] скачивание не удалось"; continue; }
  sed -e "s#simplewiki_store#enwiki_store_$tag#" -e "s#simplewiki.xml.bz2#cur.xml.bz2#" tools/exp33_dump_reader.py > /tmp/exp33_part.py
  N_ART=2000000 python3 -u /tmp/exp33_part.py > "$OUT/exp33_enwiki_$tag.log" 2>&1
  rm -f "$DATA/cur.xml.bz2"
  echo "[$tag] $(grep 'проход 2' "$OUT/exp33_enwiki_$tag.log") — $(( $(date +%s) - t0 ))s; свободно $(df -h ~ | awk 'NR==2{print $4}')"
  FREE=$(df ~ | awk 'NR==2{print $4}'); [ "$FREE" -lt 8000000 ] && { echo "мало места, стоп"; break; }
done < "$LIST"
echo "ГОТОВО"
