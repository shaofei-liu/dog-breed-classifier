#!/bin/bash
# HF Spaces 推送脚本

echo "🚀 推送到 HF Spaces"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 确保在 main 分支
git checkout main 2>/dev/null || git checkout -b main

# 添加所有文件
git add .

# 提交
git commit -m "Deploy to HF Spaces" --allow-empty

# 推送 (需要 token)
echo "准备推送到 HF Spaces..."
echo "提示: 需要输入 HF token (不是密码)"
git push -u space main

echo "✓ 推送完成!"
