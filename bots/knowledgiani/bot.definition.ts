import * as sdk from '@botpress/sdk'
import * as env from './.genenv'
import telegram from './bp_modules/telegram'

/**
 * Knowledgiani Bot Definition
 *
 * RAG logic has been moved from the knowledge plugin to src/index.ts
 * to enable deployment on workspaces that do not support plugin deployment.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * IMPLEMENTATION REVIEW (Feb 2026):
 * ─────────────────────────────────────────────────────────────────────────────────
 *
 * ✅ NO DEPENDENCY ON CUSTOM KNOWLEDGE PLUGIN
 *    - package.json: No @botpresshub/knowledge dependency
 *    - bpDependencies: Only telegram integration
 *    - bot.definition.ts: No .addPlugin(knowledge, ...) call
 *
 * ✅ BOT RUNS WITH ONLY OFFICIAL INTEGRATIONS
 *    - telegram: Official Botpress integration for messaging
 *    - No custom plugins required for deployment
 *
 * ✅ AI CALLS GO THROUGH NVIDIA API (BYOK)
 *    - src/index.ts uses direct NVIDIA API calls (OpenAI-compatible)
 *    - Primary model: meta/llama-3.1-405b-instruct
 *    - User provides their own NVIDIA API key
 *
 * ✅ RAG USES ONLY client.searchFiles()
 *    - src/index.ts: await client.searchFiles({ query })
 *    - No custom vector DB, embedding service, or RAG pipeline
 *    - Uses Botpress Cloud's built-in file search API
 *
 * NOTE: OpenAI integration removed - using NVIDIA API directly (BYOK model)
 * ═══════════════════════════════════════════════════════════════════════════════
 */

export default new sdk.BotDefinition({})
  .addIntegration(telegram, {
    enabled: true,
    configuration: {
      botToken: env.KNOWLEDGIANI_TELEGRAM_BOT_TOKEN,
    },
  })
