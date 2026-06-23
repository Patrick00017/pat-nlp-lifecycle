import { tool } from "@opencode-ai/plugin"
import path from "path"
import { $ } from "bun"

export default tool({
  description: "获取胶水的完整事件，用于进行问题的诊断",
  args: {
    start_time: tool.schema.string().describe("开始时间，ISO时间格式"),
    end_time: tool.schema.string().describe("结束时间，ISO时间格式"),
  },
  async execute(args, context) {
    const condaPath = "C:/Users/74267/.conda/envs/pat-nlp-lifecycle/python.exe"
    // const script = path.join(context.worktree, ".opencode/tools/glue_analysis.py")
    const script = 'D:/code/pat-nlp-lifecycle/app/agent_web_app/bts_analysis_sandbox/.opencode/tools/glue_analysis.py'
    
    let raw: string
    try {
      raw = await $`${condaPath} ${script} ${args.start_time} ${args.end_time}`.text()
    } catch {
      try {
        raw = await $`python ${script}`.text()
      } catch {
        try {
          raw = await $`python3 ${script}`.text()
        } catch (e) {
          return JSON.stringify({
            status: "error",
            error: "无法找到可用的 Python 解释器（尝试了 conda/python/python3）",
            detail: String(e),
          })
        }
      }
    }

    // Extract the last JSON line from mixed output (debug prints + final result)
    const lines = raw.trim().split("\n").map(l => l.trim()).filter(Boolean)
    const lastLine = lines[lines.length - 1] ?? ""
    try {
      const parsed = JSON.parse(lastLine)
      return JSON.stringify(parsed, null, 2)
    } catch {
      return raw.trim() || JSON.stringify({
        status: "error",
        error: "脚本执行成功但无有效输出",
      })
    }
  },
})