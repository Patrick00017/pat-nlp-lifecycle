import { tool } from "@opencode-ai/plugin"
import path from "path"
import { $ } from "bun"

export default tool({
  description: "获取胶水的完整事件，用于进行问题的诊断",
  args: {},
  async execute(args, context) {
    const python_path = "C:/Users/74267/.conda/envs/pat-nlp-lifecycle/python.exe"
    const script = path.join(context.worktree, ".opencode/tools/glue_analysis.py")
    const result = await $`${python_path} ${script}`.text()
    console.log(result)
    return result.trim()
  },
})