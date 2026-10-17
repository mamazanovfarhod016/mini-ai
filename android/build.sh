#!/usr/bin/env bash
# Mini AI APK (офлайн): app/www → assets/www → подписанный APK. Без Gradle.
#   ./android/build.sh            — собрать build/MiniAI.apk
# Ключ подписи и пароль лежат ВНЕ репозитория: ~/.config/mini-ai/{release.jks,keystore.txt}
set -euo pipefail
cd "$(dirname "$0")"
ROOT=..
export JAVA_HOME="${JAVA_HOME:-$HOME/opt/jdk-17.0.20.1+1}"; export PATH="$JAVA_HOME/bin:$PATH"
SDK="${ANDROID_SDK:-$HOME/opt/android-sdk}"; BT="$SDK/build-tools/34.0.0"; JAR="$SDK/platforms/android-34/android.jar"
KS="$HOME/.config/mini-ai/release.jks"; PASSF="$HOME/.config/mini-ai/keystore.txt"
[ -f "$ROOT/app/www/model.js" ] || { echo "Нет app/www/model.js — сначала: .venv/bin/python train.py"; exit 1; }
[ -f "$ROOT/app/www/model_transformer.js" ] || { echo "Нет app/www/model_transformer.js — сначала: .venv/bin/python train_transformer.py"; exit 1; }
[ -f "$KS" ] || { echo "Нет ключа $KS (см. README: keytool -genkeypair ...)"; exit 1; }
PASS="$(cat "$PASSF")"

rm -rf build; mkdir -p build/gen build/classes build/dex build/assets/www
cp -r "$ROOT/app/www/." build/assets/www/
"$BT/aapt2" compile --dir res -o build/res.zip
"$BT/aapt2" link -o build/base.apk -I "$JAR" --manifest AndroidManifest.xml -R build/res.zip -A build/assets --java build/gen --auto-add-overlay
javac -source 11 -target 11 -encoding UTF-8 -Xlint:-options -classpath "$JAR" -d build/classes $(find src build/gen -name '*.java')
"$BT/d8" --release --lib "$JAR" --min-api 21 --output build/dex $(find build/classes -name '*.class')
cp build/base.apk build/unsigned.apk
(cd build/dex && python3 -c "import zipfile;z=zipfile.ZipFile('../unsigned.apk','a');z.write('classes.dex');z.close()")
"$BT/zipalign" -f -p 4 build/unsigned.apk build/aligned.apk
"$BT/apksigner" sign --v1-signing-enabled true --v2-signing-enabled true --v3-signing-enabled true --min-sdk-version 21 \
  --ks "$KS" --ks-key-alias miniai --ks-pass "pass:$PASS" --key-pass "pass:$PASS" --out build/MiniAI.apk build/aligned.apk
"$BT/apksigner" verify build/MiniAI.apk && echo "apksigner: verified"
echo "Разрешения:"; "$BT/aapt2" dump permissions build/MiniAI.apk
if "$BT/aapt2" dump permissions build/MiniAI.apk | grep -q INTERNET; then echo "ОШИБКА: есть INTERNET"; exit 1; fi
ls -la build/MiniAI.apk
