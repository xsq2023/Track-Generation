# Contributing to Track-Generation

感谢你愿意贡献本项目！

## 贡献方式
- 提交 Bug 报告（Issue）
- 提交功能建议（Issue）
- 提交 Pull Request（文档、工程化、脚本改进）

## 开发流程
1. Fork 仓库并创建分支：`feat/xxx` 或 `fix/xxx`
2. 保持改动小而聚焦（一个 PR 做一件事）
3. 运行本地检查：
   - `ruff check .`
   - `python -m compileall scripts`
4. 更新相关文档（README / CHANGELOG）
5. 发起 PR 并按模板填写变更说明

## 代码风格
- Python 遵循 PEP 8。
- 新增脚本需支持 `--help` 参数。
- 输出文件命名保持与现有步骤一致（例如 `_batch_summary.tsv`）。

## 提交信息建议
推荐使用 Conventional Commits：
- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档变更
- `chore:` 构建/流程调整

## 行为规范
参与项目前请阅读并遵守 [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)。
