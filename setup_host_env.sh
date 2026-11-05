#!/usr/bin/env bash
set -e

TARGET_USER="${SUDO_USER:-$USER}"
TARGET_HOME=$(getent passwd "$TARGET_USER" | cut -d: -f6)
echo "Setting up development environment for user: $TARGET_USER (home: $TARGET_HOME)"

echo "=== [1/6] Configuring Java 17 ==="
JDK_DIR=$(ls -d /opt/jdk-17* 2>/dev/null | head -n1 || echo "/opt/jdk-17.0.12+7")
if [ ! -f "$JDK_DIR/bin/javac" ]; then
    echo "Downloading Eclipse Temurin JDK 17..."
    cd /tmp
    wget -q --show-progress -c https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.12%2B7/OpenJDK17U-jdk_x64_linux_hotspot_17.0.12_7.tar.gz
    mkdir -p /opt
    tar -xzf OpenJDK17U-jdk_x64_linux_hotspot_17.0.12_7.tar.gz -C /opt/
    rm -f OpenJDK17U-jdk_x64_linux_hotspot_17.0.12_7.tar.gz
    JDK_DIR=$(ls -d /opt/jdk-17* 2>/dev/null | head -n1)
fi

echo "JDK found at: $JDK_DIR"
mkdir -p /usr/lib/jvm
ln -sfn "$JDK_DIR" /usr/lib/jvm/java-17-temurin
ln -sfn "$JDK_DIR/bin/java" /usr/bin/java
ln -sfn "$JDK_DIR/bin/javac" /usr/bin/javac
ln -sfn "$JDK_DIR/bin/keytool" /usr/bin/keytool

# Fix SDKMAN symlinks in user's home
if [ -d "$TARGET_HOME/.sdkman/candidates/java" ]; then
    echo "Updating SDKMAN candidates..."
    rm -rf "$TARGET_HOME/.sdkman/candidates/java/current" "$TARGET_HOME/.sdkman/candidates/java/17.0.12-tem"
    mkdir -p "$TARGET_HOME/.sdkman/candidates/java"
    ln -s "$JDK_DIR" "$TARGET_HOME/.sdkman/candidates/java/17.0.12-tem"
    ln -s "$TARGET_HOME/.sdkman/candidates/java/17.0.12-tem" "$TARGET_HOME/.sdkman/candidates/java/current"
    chown -R "$TARGET_USER:$TARGET_USER" "$TARGET_HOME/.sdkman/candidates/java" 2>/dev/null || true
fi

export JAVA_HOME="$JDK_DIR"
export PATH="$JAVA_HOME/bin:/opt/gradle-8.9/bin:$PATH"

echo "=== [2/6] Configuring Gradle 8.9 ==="
GRADLE_DIR="/opt/gradle-8.9"
if [ ! -f "$GRADLE_DIR/bin/gradle" ]; then
    echo "Downloading Gradle 8.9..."
    cd /tmp
    wget -q --show-progress -c https://services.gradle.org/distributions/gradle-8.9-bin.zip
    unzip -q -o gradle-8.9-bin.zip -d /opt/
    rm -f gradle-8.9-bin.zip
fi
ln -sfn "$GRADLE_DIR/bin/gradle" /usr/bin/gradle

echo "=== [3/6] Setting up Android SDK (Platform 35 & Build-Tools 35.0.0) ==="
ANDROID_SDK="$TARGET_HOME/Android/Sdk"
mkdir -p "$ANDROID_SDK/cmdline-tools"

if [ ! -f "$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" ]; then
    echo "Unpacking Android Commandline Tools..."
    if [ -f /tmp/commandlinetools-linux-11076708_latest.zip ]; then
        unzip -q -o /tmp/commandlinetools-linux-11076708_latest.zip -d "$ANDROID_SDK/cmdline-tools/"
    else
        cd /tmp
        wget -q --show-progress -c https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
        unzip -q -o commandlinetools-linux-11076708_latest.zip -d "$ANDROID_SDK/cmdline-tools/"
    fi
    rm -rf "$ANDROID_SDK/cmdline-tools/latest"
    mv "$ANDROID_SDK/cmdline-tools/cmdline-tools" "$ANDROID_SDK/cmdline-tools/latest" 2>/dev/null || true
fi

export ANDROID_HOME="$ANDROID_SDK"
export PATH="$ANDROID_SDK/cmdline-tools/latest/bin:$ANDROID_SDK/platform-tools:$PATH"

echo "Accepting Android SDK licenses..."
yes | "$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" --licenses > /dev/null 2>&1 || true

echo "Installing platforms;android-35 and build-tools;35.0.0..."
"$ANDROID_SDK/cmdline-tools/latest/bin/sdkmanager" "platforms;android-35" "build-tools;35.0.0" "platform-tools"

chown -R "$TARGET_USER:$TARGET_USER" "$ANDROID_SDK" 2>/dev/null || true

# Update local.properties
PROJECT_DIR="/mnt/WindowsDrive/Shared/Projects/SIP"
if [ -f "$PROJECT_DIR/android-frontend/android/local.properties" ]; then
    sed -i '/^sdk.dir=/d' "$PROJECT_DIR/android-frontend/android/local.properties"
    echo "sdk.dir=$ANDROID_SDK" >> "$PROJECT_DIR/android-frontend/android/local.properties"
fi

echo "=== [4/6] Installing Vercel CLI & Render CLI ==="
npm install -g vercel @renderinc/cli 2>/dev/null || sudo -u "$TARGET_USER" npm install -g vercel @renderinc/cli 2>/dev/null || true

echo "=== [5/6] Checking for credentials on Raspberry Pi ==="
mkdir -p "$PROJECT_DIR/pi_backup_envs"
dnf install -y sshpass 2>/dev/null || true

for PI_HOST in raspberrypi.local 10.42.0.1 192.168.0.125; do
    echo "Trying to connect to Pi at $PI_HOST..."
    if ping -c 1 -W 2 "$PI_HOST" >/dev/null 2>&1; then
        echo "Pi is reachable at $PI_HOST! Fetching .env and keystores..."
        if [ -n "${PI_PASSWORD:-}" ]; then
            sshpass -p "$PI_PASSWORD" rsync -avz -e "ssh -o StrictHostKeyChecking=no" --include='*/' --include='*.env*' --include='*.jks' --include='*.properties' --exclude='*' "benjamin@$PI_HOST:/home/benjamin/SIP/" "$PROJECT_DIR/pi_backup_envs/" 2>/dev/null || \
            sshpass -p "$PI_PASSWORD" scp -o StrictHostKeyChecking=no -r "benjamin@$PI_HOST:/home/benjamin/SIP/android-frontend/.env*" "$PROJECT_DIR/pi_backup_envs/" 2>/dev/null || true
        else
            ssh -o StrictHostKeyChecking=no -o BatchMode=yes "benjamin@$PI_HOST" "echo Reachable via SSH key" 2>/dev/null && \
            rsync -avz -e "ssh -o StrictHostKeyChecking=no" --include='*/' --include='*.env*' --include='*.jks' --include='*.properties' --exclude='*' "benjamin@$PI_HOST:/home/benjamin/SIP/" "$PROJECT_DIR/pi_backup_envs/" 2>/dev/null || true
        fi
        break
    fi
done

echo "=== [6/6] Environment Verification ==="
"$JDK_DIR/bin/java" -version
"$JDK_DIR/bin/javac" -version
gradle --version
echo "Android SDK: $ANDROID_SDK"
echo "Setup successfully completed!"
