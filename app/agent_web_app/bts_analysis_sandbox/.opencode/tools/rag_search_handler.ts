import { tool } from "@opencode-ai/plugin"
import path from "path"
import { $ } from "bun"

export default tool({
  description: "搜索BTS产线系统文档，用于回答用户关于系统操作、功能、配置等问题。查询时使用中文关键词",
  args: {
    query: tool.schema.string().describe("搜索关键词，使用中文"),
  },
  async execute(args, context) {
    const condaPath = "C:/Users/74267/.conda/envs/pat-nlp-lifecycle/python.exe"
    const script = 'D:/code/pat-nlp-lifecycle/app/agent_web_app/bts_analysis_sandbox/.opencode/tools/rag_search.py'

    let raw
    try {
      raw = await $`${condaPath} ${script} ${args.query}`.text()
    } catch (e) {
      try {
        raw = await $`python ${script} ${args.query}`.text()
      } catch (e2) {
        return JSON.stringify({
          status: "error",
          error: "无法找到可用的 Python 解释器",
          detail: String(e2),
        })
      }
    }

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
