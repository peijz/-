#!/usr/bin/env bash
# scripts/init_db.sh
# 幂等创建 zhongyiguan 数据库。由 systemd ExecStartPre 自动调用,也可手动跑。
#
# 用法(手动):
#   sudo MYSQL_ROOT_CONF=/etc/zhongyiguan/mysql_root.conf \
#        DB_NAME=zhongyiguan \
#        /opt/zhongyiguan/scripts/init_db.sh
#
# 安全要点:
#   - MySQL 凭据走 --defaults-extra-file, 不进命令行/ps/history
#   - 脚本只 CREATE DATABASE, 不建表 (表由 app 启动时 init_db() 建)

set -euo pipefail

DB_NAME="${DB_NAME:?DB_NAME 未设置 (systemd Environment 或手动 export)}"
ROOT_CONF="${MYSQL_ROOT_CONF:-/etc/zhongyiguan/mysql_root.conf}"
MYSQL_HOST="${MYSQL_HOST:-127.0.0.1}"
MYSQL_PORT="${MYSQL_PORT:-3306}"

log()  { printf '[init_db %s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[init_db %s] FATAL: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

[[ -r "$ROOT_CONF" ]] || die "读不了 $ROOT_CONF (预期 chmod 600, owner=root)"

MYSQL=(mysql --defaults-extra-file="$ROOT_CONF"
       -h "$MYSQL_HOST"
       -P "$MYSQL_PORT"
       --protocol=TCP
       --connect-timeout=10
       --charset=utf8mb4)

# 1. 连通性测试 (失败就 fail-fast, 不让 systemd 拉起没 DB 的 app)
"${MYSQL[@]}" -e "SELECT VERSION();" >/dev/null \
    || die "MySQL 连接失败 ${MYSQL_HOST}:${MYSQL_PORT}, 请检查 $ROOT_CONF"

# 2. 幂等建库 + 字符集
SQL="CREATE DATABASE IF NOT EXISTS \`${DB_NAME}\`
     DEFAULT CHARACTER SET utf8mb4
     DEFAULT COLLATE utf8mb4_unicode_ci;"
"${MYSQL[@]}" -e "$SQL" || die "建库 SQL 执行失败"

# 3. 复核 (避免 silent failure)
FOUND=$("${MYSQL[@]}" -N -B -e \
    "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA WHERE SCHEMA_NAME='${DB_NAME}';" \
    | tr -d '[:space:]')
[[ "$FOUND" == "$DB_NAME" ]] \
    || die "建库后查不到 ${DB_NAME}, 请人工确认"

log "OK: ${DB_NAME} @ ${MYSQL_HOST}:${MYSQL_PORT} (utf8mb4 / utf8mb4_unicode_ci)"