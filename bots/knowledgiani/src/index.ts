/**
 * Knowledgiani Bot - RAG-enabled knowledge assistant
 *
 * This bot implements RAG (Retrieval-Augmented Generation) logic that was
 * previously in the knowledge plugin (plugins/knowledge/src/index.ts).
 *
 * The logic has been moved here to enable deployment on workspaces that
 * do not support plugin deployment.
 *
 * LLM Provider: NVIDIA API (OpenAI-compatible)
 * Primary Model: meta/llama-3.1-405b-instruct
 */

import { z } from '@botpress/sdk'
import JSON5 from 'json5'
import { jsonrepair } from 'jsonrepair'
import * as bp from '.botpress'
import * as env from '.genenv'

// ═══════════════════════════════════════════════════════════════════════════════
// NVIDIA API CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

const NVIDIA_CONFIG = {
  baseUrl: 'https://integrate.api.nvidia.com/v1',
  apiKey: env.NVIDIA_API_KEY,
  model: 'meta/llama-3.1-405b-instruct',
  maxTokens: 2048,
}

// ═══════════════════════════════════════════════════════════════════════════════
// LLM HELPER TYPES AND FUNCTIONS
// (previously in plugins/knowledge/src/generate-content.ts)
// ═══════════════════════════════════════════════════════════════════════════════

type LLMInput = {
  responseFormat?: 'json_object' | 'text'
  temperature?: number
  systemPrompt: string
  messages: Array<{ role: 'user' | 'assistant'; content: string }>
}

type LLMOutput = {
  choices: Array<{ content: string }>
}

const tryParseJson = (str: string): object | string => {
  try {
    return JSON5.parse(jsonrepair(str))
  } catch {
    return str
  }
}

const parseLLMOutput = (output: LLMOutput): { success: boolean; json: object } => {
  const firstChoice = output.choices[0]?.content
  if (!firstChoice) {
    return { success: false, json: {} }
  }
  return {
    success: true,
    json: tryParseJson(firstChoice) as object,
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// QUESTION EXTRACTION SCHEMAS AND PROMPTS
// (previously in plugins/knowledge/src/question-prompt.ts)
// ═══════════════════════════════════════════════════════════════════════════════

const ExtractedQuestion = z.object({
  line: z.string().describe('The line number of the question (must be prefixed with "L")'),
  raw_question: z.string().describe('The raw question extracted from the user message'),
  resolved_question: z.string().describe('The resolved question with any missing context filled in'),
  search_query: z.string().describe('The search query that would be used to find the answer in a search engine'),
})

const QuestionOutputFormat = z.object({
  hasQuestions: z.boolean().describe('Whether or not questions were found in the user message'),
  questions: z
    .array(ExtractedQuestion)
    .describe('List of extracted questions, or an empty array if no questions are found')
    .optional(),
})

const createQuestionExtractionPrompt = (text: string): LLMInput => ({
  responseFormat: 'json_object',
  temperature: 0,
  systemPrompt: `
You are a question extractor.
You will be given a USER MESSAGE.
Your goal is to respond with the a list of question(s) found in the USER MESSAGE, if any.
For the purpose of this task, a question is defined as any sentence that is asking for information or is seeking an answer.

Include the raw_question (with no modifications), the resolved_question (with any missing context filled in), and the search_query (the query that would be used to find the answer in a search engine).

If there are no questions in the USER MESSAGE, return an empty array.

Always respond in JSON with the following format:
{
  "hasQuestions": boolean,
  "questions": [
    {
      "line": "L1",
      "raw_question": "the original question",
      "resolved_question": "the question with context filled in",
      "search_query": "optimized search query"
    }
  ]
}
`.trim(),
  messages: [
    {
      role: 'user',
      content: `<USER MESSAGE>\n[L1]\t ${text}\n</USER MESSAGE>`,
    },
  ],
})

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIDENCE CONFIGURATION
// (previously in plugins/knowledge/src/index.ts - CONFIDENCE_CONFIG)
//
// THRESHOLD RATIONALE (tuned from production traffic analysis, Feb 2026):
// These values are intentionally conservative to reduce false positives.
// ═══════════════════════════════════════════════════════════════════════════════

const CONFIDENCE_CONFIG = {
  minPassages: 1,
  minContentLength: 50,
  minPassageScore: 0.55,
  lowScoreThreshold: 0.65,
  maxDuplicateRatio: 0.6,
  minQueryLength: 4,
  lowConfidenceFallback:
    "I don't have enough information in my knowledge base to answer that question confidently. Could you try rephrasing or asking something else?",
}

// ═══════════════════════════════════════════════════════════════════════════════
// STOP WORDS FOR QUERY QUALITY CHECK
// (previously in plugins/knowledge/src/index.ts)
// ═══════════════════════════════════════════════════════════════════════════════

const STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'it', 'to', 'of', 'and', 'or', 'for', 'in', 'on', 'at',
  'be', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
  'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
  'what', 'which', 'who', 'how', 'why', 'when', 'where', 'there', 'here', 'with',
  'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
  'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
  'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them', 'its',
])

// ═══════════════════════════════════════════════════════════════════════════════
// QUERY QUALITY EVALUATION
// (previously in plugins/knowledge/src/index.ts - evaluateQueryQuality)
// ═══════════════════════════════════════════════════════════════════════════════

const evaluateQueryQuality = (query: string): { isWeak: boolean; reason: string } => {
  const trimmed = query.trim()

  if (!trimmed) {
    return { isWeak: true, reason: 'empty_query' }
  }

  if (trimmed.length < CONFIDENCE_CONFIG.minQueryLength) {
    return { isWeak: true, reason: 'query_too_short' }
  }

  const tokens = trimmed.toLowerCase().split(/\s+/).filter(Boolean)
  if (tokens.length === 0) {
    return { isWeak: true, reason: 'no_tokens' }
  }

  const nonStopTokens = tokens.filter((t) => !STOP_WORDS.has(t))

  if (nonStopTokens.length === 0) {
    return { isWeak: true, reason: 'only_stop_words' }
  }

  const meaningfulRatio = nonStopTokens.length / tokens.length
  if (meaningfulRatio < 0.2) {
    return { isWeak: true, reason: 'mostly_stop_words' }
  }

  return { isWeak: false, reason: 'ok' }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTI-TURN CONTEXT HANDLING
// (previously in plugins/knowledge/src/index.ts)
// ═══════════════════════════════════════════════════════════════════════════════

const CONTEXT_CONFIG = {
  maxContextMessages: 6,
  shortMessageThreshold: 50,
  referentialPatterns: [
    /^(and|but|also|what about|how about|tell me about|explain|more on|that|this|it|they|those|these)\b/i,
    /\?$/,
    /^(pricing|refunds?|policy|details?|features?|costs?|plans?)\??$/i,
  ],
}

const StandaloneQuestionSchema = z.object({
  needsContext: z.boolean().describe('True if the message references prior conversation'),
  standaloneQuestion: z.string().describe('A clear, self-contained question'),
  searchQuery: z.string().describe('An optimized search query'),
  isValidQuestion: z.boolean().describe('True if this is a valid question'),
})

const AnswerOutputSchema = z.object({
  answer: z.string().describe('The answer based ONLY on the provided passages'),
  hasEnoughInfo: z.boolean().describe('True if passages contain enough information'),
  confidence: z.enum(['high', 'medium', 'low']).describe('Confidence in the answer'),
})

const isLikelyReferential = (text: string): boolean => {
  const trimmed = text.trim()
  if (trimmed.length < CONTEXT_CONFIG.shortMessageThreshold) {
    return true
  }
  return CONTEXT_CONFIG.referentialPatterns.some(pattern => pattern.test(trimmed))
}

const formatConversationContext = (
  messages: Array<{ type: string; payload: unknown; userId: string }>,
  botId: string
): string => {
  if (!messages.length) {
    return ''
  }

  const lines = messages.map((msg) => {
    const role = msg.userId === botId ? 'bot' : 'user'
    const text = msg.type === 'text' && msg.payload && typeof msg.payload === 'object' && 'text' in msg.payload
      ? String((msg.payload as { text: string }).text)
      : '[non-text message]'
    return `${role}: ${text}`
  })

  return `Transcript:\n"""\n${lines.join('\n')}\n"""`
}

const createStandaloneQuestionPrompt = (
  currentMessage: string,
  conversationContext: string
): LLMInput => ({
  responseFormat: 'json_object',
  temperature: 0,
  systemPrompt: `You are a question reconstruction assistant. Your task is to take a user message that may be referential or incomplete and reconstruct it into a clear, standalone question using the conversation context.

RULES:
1. If the message references something from prior conversation, combine them into a single explicit question
2. If the message is already complete, return it as-is with needsContext=false
3. The standaloneQuestion should be understandable WITHOUT reading prior conversation
4. Generate an optimized searchQuery for finding relevant documents
5. Set isValidQuestion=false only if the input is not seeking information

Respond with JSON: { needsContext: boolean, standaloneQuestion: string, searchQuery: string, isValidQuestion: boolean }`,
  messages: [
    {
      role: 'user',
      content: `CONVERSATION CONTEXT:\n${conversationContext || 'No prior context.'}\n\nCURRENT USER MESSAGE: "${currentMessage}"\n\nReconstruct this into a standalone question if needed.`,
    },
  ],
})

// ═══════════════════════════════════════════════════════════════════════════════
// PASSAGE TYPES AND CONFIDENCE CALCULATIONS
// (previously in plugins/knowledge/src/index.ts)
// ═══════════════════════════════════════════════════════════════════════════════

type Passage = {
  content: string
  score: number
  file: { id: string; key: string }
  meta?: { type?: string; subtype?: string }
}

type RetrievalConfidenceResult = {
  isConfident: boolean
  reason: string
  signals: {
    avgScore: number
    duplicateRatio: number
    highDuplication: boolean
    highContentSimilarity: boolean
    highQualityCount: number
    totalContentLength: number
  }
}

const calculateDuplicationRatio = (passages: Passage[]): number => {
  if (passages.length <= 1) {
    return 0
  }
  const uniqueFileKeys = new Set(passages.map((p) => p.file.key))
  return 1 - uniqueFileKeys.size / passages.length
}

const hasHighContentSimilarity = (passages: Passage[]): boolean => {
  if (passages.length <= 1) {
    return false
  }

  const wordSets = passages.map((p) => {
    const words = p.content.toLowerCase().split(/\s+/).filter((w) => w.length > 3)
    return new Set(words)
  })

  const checkCount = Math.min(3, wordSets.length)
  let highOverlapCount = 0

  for (let i = 0; i < checkCount; i++) {
    for (let j = i + 1; j < checkCount; j++) {
      const setA = wordSets[i]!
      const setB = wordSets[j]!
      if (setA.size === 0 || setB.size === 0) continue

      const intersection = [...setA].filter((w) => setB.has(w)).length
      const union = new Set([...setA, ...setB]).size
      const jaccard = intersection / union

      if (jaccard > 0.6) {
        highOverlapCount++
      }
    }
  }

  const totalPairs = (checkCount * (checkCount - 1)) / 2
  return highOverlapCount >= totalPairs * 0.5
}

const calculateRetrievalConfidence = (passages: Passage[]): RetrievalConfidenceResult => {
  const signals = {
    avgScore: 0,
    duplicateRatio: 0,
    highDuplication: false,
    highContentSimilarity: false,
    highQualityCount: 0,
    totalContentLength: 0,
  }

  if (passages.length < CONFIDENCE_CONFIG.minPassages) {
    return { isConfident: false, reason: 'insufficient_passages', signals }
  }

  signals.totalContentLength = passages.reduce((sum, p) => sum + p.content.length, 0)
  if (signals.totalContentLength < CONFIDENCE_CONFIG.minContentLength) {
    return { isConfident: false, reason: 'insufficient_content', signals }
  }

  const scores = passages.map((p) => p.score)
  signals.avgScore = scores.reduce((a, b) => a + b, 0) / scores.length
  signals.highQualityCount = passages.filter(
    (p) => p.score >= CONFIDENCE_CONFIG.minPassageScore
  ).length

  if (signals.highQualityCount === 0) {
    return { isConfident: false, reason: 'no_high_quality_passages', signals }
  }

  signals.duplicateRatio = calculateDuplicationRatio(passages)
  signals.highDuplication = signals.duplicateRatio > CONFIDENCE_CONFIG.maxDuplicateRatio
  signals.highContentSimilarity = hasHighContentSimilarity(passages)

  if (signals.avgScore < CONFIDENCE_CONFIG.lowScoreThreshold) {
    return { isConfident: false, reason: 'low_average_score', signals }
  }

  if (passages.length <= 2 && signals.highDuplication) {
    return { isConfident: false, reason: 'few_passages_high_duplication', signals }
  }

  return { isConfident: true, reason: 'ok', signals }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANSWER GENERATION PROMPT
// (previously in plugins/knowledge/src/index.ts - createAnswerPrompt)
// ═══════════════════════════════════════════════════════════════════════════════

const createAnswerPrompt = (question: string, passages: string[]): LLMInput => ({
  responseFormat: 'json_object',
  temperature: 0.1,
  systemPrompt: `You are a helpful assistant that answers questions using ONLY the information provided in the passages below.

CRITICAL RULES:
1. Answer ONLY based on information explicitly stated in the passages
2. Do NOT make up or infer information not present in the passages
3. If the passages do not contain enough information, set hasEnoughInfo to false
4. Be concise and direct in your answers
5. If multiple passages contain relevant information, synthesize them

Respond with JSON: { answer: string, hasEnoughInfo: boolean, confidence: "high" | "medium" | "low" }`,
  messages: [
    {
      role: 'user',
      content: `PASSAGES:\n${passages.map((p, i) => `[${i + 1}] ${p}`).join('\n\n')}\n\nQUESTION: ${question}\n\nPlease answer using ONLY the information in the passages above.`,
    },
  ],
})

// ═══════════════════════════════════════════════════════════════════════════════
// BOT IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

const bot = new bp.Bot({
  actions: {},
})

/**
 * Helper to call NVIDIA API (OpenAI-compatible)
 * Primary model: meta/llama-3.1-405b-instruct
 */
const callLLM = async (
  _client: bp.Client,
  input: LLMInput
): Promise<LLMOutput> => {
  const messages = [
    { role: 'system' as const, content: input.systemPrompt },
    ...input.messages.map((m) => ({
      role: m.role as 'user' | 'assistant',
      content: m.content,
    })),
  ]

  const response = await fetch(`${NVIDIA_CONFIG.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${NVIDIA_CONFIG.apiKey}`,
    },
    body: JSON.stringify({
      model: NVIDIA_CONFIG.model,
      messages,
      max_tokens: NVIDIA_CONFIG.maxTokens,
      temperature: input.temperature ?? 0.7,
      ...(input.responseFormat === 'json_object' && {
        response_format: { type: 'json_object' },
      }),
    }),
  })

  if (!response.ok) {
    const error = await response.text()
    console.error('NVIDIA API error:', response.status, error)
    throw new Error(`NVIDIA API error: ${response.status} ${error}`)
  }

  const data = (await response.json()) as {
    choices: Array<{
      message: {
        content: string | null
        reasoning_content?: string
      }
    }>
  }

  return {
    choices: data.choices.map((c) => ({
      content: c.message.content ?? c.message.reasoning_content ?? '',
    })),
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MESSAGE HANDLER - RAG LOGIC
// (previously in plugins/knowledge/src/index.ts - plugin.on.beforeIncomingMessage)
// ═══════════════════════════════════════════════════════════════════════════════

bot.on.message('text', async (props) => {
  const { message, client, ctx } = props
  const text: string = message.payload.text

  if (!text) {
    console.debug('Ignoring empty message')
    return
  }

  console.debug('Processing message:', text)

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 1: FETCH CONVERSATION CONTEXT (for multi-turn understanding)
  // This was previously in the knowledge plugin
  // ─────────────────────────────────────────────────────────────────────────────
  let conversationContext = ''
  let recentMessages: Array<{ type: string; payload: unknown; userId: string }> = []

  try {
    const { messages: historyMessages } = await client.listMessages({
      conversationId: message.conversationId,
      tags: {},
    })

    recentMessages = historyMessages
      .filter((m) => m.id !== message.id)
      .slice(0, CONTEXT_CONFIG.maxContextMessages)
      .reverse()

    conversationContext = formatConversationContext(recentMessages, ctx.botId)
    console.debug('Fetched conversation context:', recentMessages.length, 'messages')
  } catch (err) {
    console.debug('Could not fetch conversation history:', err)
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 2: MULTI-TURN QUESTION RECONSTRUCTION
  // This was previously in the knowledge plugin
  // ─────────────────────────────────────────────────────────────────────────────
  let standaloneQuestion = text
  let searchQuery = text
  let skipRegularExtraction = false

  if (isLikelyReferential(text) && conversationContext) {
    console.debug('Message appears referential, reconstructing standalone question...')

    try {
      const reconstructPrompt = createStandaloneQuestionPrompt(text, conversationContext)
      const reconstructOutput = await callLLM(client, reconstructPrompt)

      const { success: reconstructSuccess, json: reconstructJson } = parseLLMOutput(reconstructOutput)
      if (reconstructSuccess) {
        const reconstructResult = StandaloneQuestionSchema.safeParse(reconstructJson)
        if (reconstructResult.success) {
          const { standaloneQuestion: reconstructed, searchQuery: query, isValidQuestion, needsContext } = reconstructResult.data

          if (!isValidQuestion) {
            console.debug('Not a valid question after reconstruction, responding with greeting')
            await client.createMessage({
              conversationId: message.conversationId,
              userId: ctx.botId,
              payload: { text: "Hello! I'm here to help answer questions based on my knowledge base. What would you like to know?" },
              tags: {},
              type: 'text',
            })
            return
          }

          standaloneQuestion = reconstructed
          searchQuery = query || reconstructed

          if (needsContext) {
            skipRegularExtraction = true
            console.debug('Reconstructed standalone question:', standaloneQuestion)
            console.debug('Search query:', searchQuery)
          }
        }
      }
    } catch (err) {
      console.debug('Question reconstruction failed, falling back to regular extraction:', err)
    }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 3: REGULAR QUESTION EXTRACTION (fallback path)
  // This was previously in the knowledge plugin
  // ─────────────────────────────────────────────────────────────────────────────
  if (!skipRegularExtraction) {
    const llmInput = createQuestionExtractionPrompt(text)

    if (conversationContext && llmInput.messages.length > 0) {
      const lastMessage = llmInput.messages[llmInput.messages.length - 1]
      if (lastMessage && lastMessage.role === 'user') {
        lastMessage.content = `${conversationContext}\n\n${lastMessage.content}`
      }
    }

    const llmOutput = await callLLM(client, llmInput)

    const { success, json } = parseLLMOutput(llmOutput)
    if (!success) {
      console.debug('Failed to extract questions, sending fallback')
      await client.createMessage({
        conversationId: message.conversationId,
        userId: ctx.botId,
        payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
        tags: {},
        type: 'text',
      })
      return
    }

    const parsedResult = QuestionOutputFormat.safeParse(json)
    if (!parsedResult.success) {
      console.debug('Failed to parse question format, sending fallback')
      await client.createMessage({
        conversationId: message.conversationId,
        userId: ctx.botId,
        payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
        tags: {},
        type: 'text',
      })
      return
    }

    const { data } = parsedResult
    if (!data.hasQuestions || !data.questions?.length) {
      console.debug('No questions extracted, responding with greeting')
      await client.createMessage({
        conversationId: message.conversationId,
        userId: ctx.botId,
        payload: { text: "Hello! I'm here to help answer questions based on my knowledge base. What would you like to know?" },
        tags: {},
        type: 'text',
      })
      return
    }

    searchQuery = data.questions.map((q) => q.search_query || q.resolved_question).join(' ')
    standaloneQuestion = data.questions.map((q) => q.resolved_question).join(' ')

    console.debug('Extracted search query:', searchQuery)
    console.debug('Resolved question:', standaloneQuestion)
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 3.5: QUERY QUALITY CHECK
  // This was previously in the knowledge plugin
  // ─────────────────────────────────────────────────────────────────────────────
  const queryQuality = evaluateQueryQuality(searchQuery)
  if (queryQuality.isWeak) {
    console.debug('Query quality too weak for reliable retrieval:', queryQuality.reason)
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
      tags: {},
      type: 'text',
    })
    return
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 4: RETRIEVAL using client.searchFiles()
  // This was previously in the knowledge plugin - ONLY allowed RAG pattern
  // ─────────────────────────────────────────────────────────────────────────────
  console.debug('Searching for:', searchQuery)
  const { passages } = await client.searchFiles({ query: searchQuery })

  if (!passages.length) {
    console.debug('No passages found')
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
      tags: {},
      type: 'text',
    })
    return
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 5: CONFIDENCE HANDLING (Retrieval Quality)
  // This was previously in the knowledge plugin
  // ─────────────────────────────────────────────────────────────────────────────
  const retrievalConfidence = calculateRetrievalConfidence(passages)

  console.debug('Retrieval confidence:', {
    isConfident: retrievalConfidence.isConfident,
    reason: retrievalConfidence.reason,
    avgScore: retrievalConfidence.signals.avgScore.toFixed(3),
    highQualityCount: retrievalConfidence.signals.highQualityCount,
    highDuplication: retrievalConfidence.signals.highDuplication,
    highContentSimilarity: retrievalConfidence.signals.highContentSimilarity,
  })

  if (!retrievalConfidence.isConfident) {
    console.debug('Low confidence in retrieval:', retrievalConfidence.reason)
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
      tags: {},
      type: 'text',
    })
    return
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 6: ANSWER GENERATION
  // This was previously in the knowledge plugin
  // ─────────────────────────────────────────────────────────────────────────────
  const passageContents = passages.map((p) => p.content)

  try {
    const answerPrompt = createAnswerPrompt(standaloneQuestion, passageContents)
    const answerOutput = await callLLM(client, answerPrompt)

    const { success: answerSuccess, json: answerJson } = parseLLMOutput(answerOutput)
    if (!answerSuccess) {
      console.debug('Failed to parse answer output')
      throw new Error('Failed to parse answer')
    }

    const answerResult = AnswerOutputSchema.safeParse(answerJson)
    if (!answerResult.success) {
      console.debug('Answer output does not match schema:', answerResult.error)
      throw new Error('Answer schema validation failed')
    }

    const { answer, hasEnoughInfo, confidence: answerConfidence } = answerResult.data

    // ─────────────────────────────────────────────────────────────────────────────
    // STEP 7: FINAL GATING DECISION
    // This was previously in the knowledge plugin
    // ─────────────────────────────────────────────────────────────────────────────
    const { signals } = retrievalConfidence

    // Case 1: Model explicitly says it doesn't have enough info
    if (!hasEnoughInfo) {
      console.debug('Model indicates insufficient information in passages')
      await client.createMessage({
        conversationId: message.conversationId,
        userId: ctx.botId,
        payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
        tags: {},
        type: 'text',
      })
      return
    }

    // Case 2: Low answer confidence from the model
    if (answerConfidence === 'low') {
      console.debug('Answer confidence is low')
      await client.createMessage({
        conversationId: message.conversationId,
        userId: ctx.botId,
        payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
        tags: {},
        type: 'text',
      })
      return
    }

    // Case 3: Medium confidence with weak retrieval signals = reject
    if (answerConfidence === 'medium') {
      const hasWeakSignals =
        signals.highDuplication ||
        signals.highContentSimilarity ||
        signals.highQualityCount <= 1

      if (hasWeakSignals) {
        console.debug('Medium confidence with weak retrieval signals, rejecting answer')
        await client.createMessage({
          conversationId: message.conversationId,
          userId: ctx.botId,
          payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
          tags: {},
          type: 'text',
        })
        return
      }
    }

    // Send the generated answer
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: { text: answer },
      tags: {},
      type: 'text',
    })
  } catch (err) {
    console.error('Answer generation failed:', err)
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: { text: CONFIDENCE_CONFIG.lowConfidenceFallback },
      tags: {},
      type: 'text',
    })
  }
})

// ═══════════════════════════════════════════════════════════════════════════════
// FILE UPLOAD HANDLER (original bot functionality)
// ═══════════════════════════════════════════════════════════════════════════════

const fileKey = (url: string) => {
  const fileName = url.split('/').pop()
  if (!fileName) {
    return url
  }
  return fileName
}

bot.on.message('file', async (props) => {
  console.info('Received file message:', props.message.payload.fileUrl)

  const { fileUrl } = props.message.payload
  const key = fileKey(fileUrl)
  await props.client.uploadFile({
    key,
    url: fileUrl,
    index: true,
  })

  console.info('File uploaded:', key)
})

export default bot
