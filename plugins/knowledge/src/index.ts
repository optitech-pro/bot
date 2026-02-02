import { z } from '@botpress/sdk'
import * as gen from './generate-content'
import * as questions from './question-prompt'
import * as bp from '.botpress'

/**
 * Configuration for confidence thresholds and fallback behavior.
 * Low confidence threshold determines when to return a fallback response.
 *
 * ═══════════════════════════════════════════════════════════════════════════════
 * THRESHOLD RATIONALE (tuned from production traffic analysis, Feb 2026):
 * ─────────────────────────────────────────────────────────────────────────────────
 * These values are intentionally conservative to reduce false positives (weak
 * or incorrect answers). It's safer to return a fallback than to provide an
 * uncertain answer. The thresholds were calibrated based on observed signal
 * distributions in real traffic, prioritizing precision over recall.
 *
 * Key observations that informed these values:
 *   - Passages with score < 0.55 often produced incorrect or tangential answers
 *   - Average scores below 0.65 correlated with user-reported bad answers
 *   - High duplication (>60% from same source) indicated narrow/weak retrieval
 *   - Queries under 4 chars frequently matched irrelevant content
 *
 * STABLE thresholds (rarely need adjustment):
 *   - minPassages: 1 is correct for most use cases
 *   - minContentLength: 50 chars ensures minimum viable context
 * ═══════════════════════════════════════════════════════════════════════════════
 */
const CONFIDENCE_CONFIG = {
  /**
   * Minimum number of passages required for a confident answer.
   * [STABLE] Rarely needs adjustment.
   */
  minPassages: 1,

  /**
   * Minimum character length of combined passage content for confidence.
   * [STABLE] Rarely needs adjustment.
   */
  minContentLength: 50,

  /**
   * Minimum score threshold for a passage to be considered "high quality".
   * Score is the similarity score returned by searchFiles (higher = more similar).
   * Raised from 0.5 to 0.55 to reduce false positives from marginal matches.
   */
  minPassageScore: 0.55,

  /**
   * If the average passage score is below this threshold, reduce confidence.
   * This indicates the retrieved passages may not be highly relevant.
   * Raised from 0.6 to 0.65 based on observed correlation with answer quality.
   */
  lowScoreThreshold: 0.65,

  /**
   * Maximum ratio of duplicates allowed before down-ranking confidence.
   * A high duplicate ratio (e.g., same file/source repeated) often indicates
   * weak retrieval or a query that only matches limited content.
   * Lowered from 0.7 to 0.6 to require better source diversity.
   */
  maxDuplicateRatio: 0.6,

  /**
   * Minimum length for a search query to be considered valid.
   * Very short queries often result in poor retrieval.
   * Raised from 3 to 4 to filter more noise while still allowing
   * legitimate short queries (e.g., "API", "auth").
   */
  minQueryLength: 4,

  /** Fallback message when confidence is too low to provide an answer */
  lowConfidenceFallback:
    "I don't have enough information in my knowledge base to answer that question confidently. Could you try rephrasing or asking something else?",
}

/**
 * Common stop words that don't carry semantic meaning for search.
 * If a query consists mostly of these, it's likely too weak.
 */
const STOP_WORDS = new Set([
  'a', 'an', 'the', 'is', 'it', 'to', 'of', 'and', 'or', 'for', 'in', 'on', 'at',
  'be', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
  'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
  'what', 'which', 'who', 'how', 'why', 'when', 'where', 'there', 'here', 'with',
  'from', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
  'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
  'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them', 'its',
])

/**
 * Evaluates query quality to determine if it's likely to produce good retrieval.
 * Returns an object with quality assessment and reason.
 *
 * A weak query (too short, empty, or mostly stop words) will likely produce
 * poor retrieval results, so we should be less confident in those cases.
 */
const evaluateQueryQuality = (
  query: string
): { isWeak: boolean; reason: string } => {
  const trimmed = query.trim()

  // Empty query is definitely weak
  if (!trimmed) {
    return { isWeak: true, reason: 'empty_query' }
  }

  // Query too short (less than minQueryLength chars)
  if (trimmed.length < CONFIDENCE_CONFIG.minQueryLength) {
    return { isWeak: true, reason: 'query_too_short' }
  }

  // Tokenize and check if query is mostly stop words
  const tokens = trimmed.toLowerCase().split(/\s+/).filter(Boolean)
  if (tokens.length === 0) {
    return { isWeak: true, reason: 'no_tokens' }
  }

  const nonStopTokens = tokens.filter((t) => !STOP_WORDS.has(t))

  // If all tokens are stop words, query is weak
  if (nonStopTokens.length === 0) {
    return { isWeak: true, reason: 'only_stop_words' }
  }

  // If less than 20% of tokens are meaningful, query is weak
  const meaningfulRatio = nonStopTokens.length / tokens.length
  if (meaningfulRatio < 0.2) {
    return { isWeak: true, reason: 'mostly_stop_words' }
  }

  return { isWeak: false, reason: 'ok' }
}

/**
 * Configuration for multi-turn context handling.
 */
const CONTEXT_CONFIG = {
  /** Maximum number of recent messages to fetch for context */
  maxContextMessages: 6,
  /** Character threshold below which a message is considered "short" and likely referential */
  shortMessageThreshold: 50,
  /** Patterns that indicate a follow-up/referential message */
  referentialPatterns: [
    /^(and|but|also|what about|how about|tell me about|explain|more on|that|this|it|they|those|these)\b/i,
    /\?$/,  // Single word/short phrase ending in question mark
    /^(pricing|refunds?|policy|details?|features?|costs?|plans?)\??$/i,  // Single topic words
  ],
}

/**
 * Schema for reconstructing a standalone question from a referential/follow-up message.
 * Used when the user message is short or contains references to prior context.
 */
const StandaloneQuestionSchema = z.object({
  /** Whether the current message requires context from previous messages to understand */
  needsContext: z.boolean().describe('True if the message references prior conversation and needs context to understand'),
  /** The reconstructed standalone question that can be understood without prior context */
  standaloneQuestion: z.string().describe('A clear, self-contained question that incorporates any necessary context from the conversation'),
  /** Optimized search query for retrieval */
  searchQuery: z.string().describe('An optimized search query for finding relevant documents in a knowledge base'),
  /** Whether this is a valid question worth searching for */
  isValidQuestion: z.boolean().describe('True if this is a valid question that requires a knowledge lookup'),
})

/**
 * Schema for the answer generation output.
 * Instructs the LLM to answer from passages only and indicate confidence.
 */
const AnswerOutputSchema = z.object({
  /** The generated answer based solely on the passages */
  answer: z.string().describe('The answer to the question based ONLY on the provided passages'),
  /** Whether the passages contain sufficient information to answer */
  hasEnoughInfo: z.boolean().describe('True if the passages contain enough information to answer the question'),
  /** Confidence level: high, medium, or low */
  confidence: z.enum(['high', 'medium', 'low']).describe('Confidence in the answer based on passage relevance'),
})

/**
 * Determines if a message is likely referential/short and may need context.
 * Checks message length and common follow-up patterns.
 */
const isLikelyReferential = (text: string): boolean => {
  const trimmed = text.trim()
  
  // Short messages are likely referential
  if (trimmed.length < CONTEXT_CONFIG.shortMessageThreshold) {
    return true
  }
  
  // Check for common follow-up patterns
  return CONTEXT_CONFIG.referentialPatterns.some(pattern => pattern.test(trimmed))
}

/**
 * Formats conversation history into a context string for the LLM.
 * Takes recent messages and formats them as a readable transcript.
 */
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

/**
 * Creates a prompt to reconstruct a standalone question from a referential message.
 * Combines the current message with conversation context.
 */
const createStandaloneQuestionPrompt = (
  currentMessage: string,
  conversationContext: string
): gen.LLMInput => ({
  responseFormat: 'json_object',
  temperature: 0,
  systemPrompt: `You are a question reconstruction assistant. Your task is to take a user message that may be referential or incomplete (e.g., "what about pricing?", "and refunds?", "that policy?") and reconstruct it into a clear, standalone question using the conversation context.

RULES:
1. If the message clearly references something from the prior conversation, combine them into a single explicit question
2. If the message is already a complete, standalone question, return it as-is with needsContext=false
3. The standaloneQuestion should be understandable WITHOUT reading the prior conversation
4. Generate an optimized searchQuery for finding relevant documents
5. Set isValidQuestion=false only if the input is not seeking information (e.g., greetings, acknowledgments)

You must respond with JSON matching this schema:
${StandaloneQuestionSchema.toTypescriptType({ treatDefaultAsOptional: true })}`,
  messages: [
    {
      role: 'user',
      content: `CONVERSATION CONTEXT:
${conversationContext || 'No prior context available.'}

CURRENT USER MESSAGE: "${currentMessage}"

Reconstruct this into a standalone question if needed.`,
    },
  ],
})

/**
 * Passage type representing the structure returned by client.searchFiles().
 * Includes content, score (similarity), and file metadata.
 */
type Passage = {
  content: string
  /** Similarity score from vector search - higher is more relevant */
  score: number
  file: {
    id: string
    key: string
  }
  meta?: {
    type?: string
    subtype?: string
  }
}

/**
 * Calculates the duplication ratio of retrieved passages.
 * High duplication (same file/source repeated) often indicates weak retrieval
 * where the query only matches limited content.
 *
 * Returns a value between 0 and 1, where 1 means all passages are from the same source.
 */
const calculateDuplicationRatio = (passages: Passage[]): number => {
  if (passages.length <= 1) {
    return 0 // Single passage cannot be duplicative
  }

  // Count unique file keys
  const uniqueFileKeys = new Set(passages.map((p) => p.file.key))
  const uniqueRatio = uniqueFileKeys.size / passages.length

  // Duplication ratio is the inverse of unique ratio
  // E.g., if 5 passages from 1 file: unique=1/5=0.2, duplicate=0.8
  return 1 - uniqueRatio
}

/**
 * Checks if passages have highly similar content (near-duplicates).
 * Uses a simple Jaccard-like overlap heuristic on word sets.
 */
const hasHighContentSimilarity = (passages: Passage[]): boolean => {
  if (passages.length <= 1) {
    return false
  }

  // Create word sets for each passage
  const wordSets = passages.map((p) => {
    const words = p.content.toLowerCase().split(/\s+/).filter((w) => w.length > 3)
    return new Set(words)
  })

  // Check pairwise overlap for first few passages
  const checkCount = Math.min(3, wordSets.length)
  let highOverlapCount = 0

  for (let i = 0; i < checkCount; i++) {
    for (let j = i + 1; j < checkCount; j++) {
      const setA = wordSets[i]!
      const setB = wordSets[j]!
      if (setA.size === 0 || setB.size === 0) continue

      // Calculate Jaccard similarity
      const intersection = [...setA].filter((w) => setB.has(w)).length
      const union = new Set([...setA, ...setB]).size
      const jaccard = intersection / union

      if (jaccard > 0.6) {
        highOverlapCount++
      }
    }
  }

  // If most pairs have high overlap, content is duplicative
  const totalPairs = (checkCount * (checkCount - 1)) / 2
  return highOverlapCount >= totalPairs * 0.5
}

/**
 * Detailed retrieval confidence result with multiple signals.
 * These signals are used for both gating decisions and diagnostic logging.
 */
type RetrievalConfidenceResult = {
  /** Whether overall retrieval confidence is sufficient */
  isConfident: boolean
  /** Primary reason for the confidence decision */
  reason: string
  /** Additional signals for stricter gating decisions and diagnostic logging */
  signals: {
    /** Average relevance score of passages */
    avgScore: number
    /** Raw duplication ratio (0-1, higher = more duplicates). Used for logging. */
    duplicateRatio: number
    /** Whether passages are highly duplicative (same source) */
    highDuplication: boolean
    /** Whether passage content is very similar (near-duplicates) */
    highContentSimilarity: boolean
    /** Number of high-quality passages (above score threshold) */
    highQualityCount: number
    /** Total content length in characters. Used for logging. */
    totalContentLength: number
  }
}

/**
 * Calculates a confidence score based on retrieval results.
 *
 * This function evaluates multiple signals:
 * 1. Passage count: Need at least minPassages to answer
 * 2. Content length: Need sufficient text content
 * 3. Relevance scores: Uses the score field from searchFiles to assess quality
 * 4. Duplication: Down-ranks if passages are from the same source
 * 5. Content similarity: Detects near-duplicate content
 *
 * These signals help identify cases where retrieval may have failed or
 * where the query only matched limited, repetitive content.
 */
const calculateRetrievalConfidence = (
  passages: Passage[]
): RetrievalConfidenceResult => {
  // Initialize signals with all fields for diagnostic logging
  const signals = {
    avgScore: 0,
    duplicateRatio: 0,
    highDuplication: false,
    highContentSimilarity: false,
    highQualityCount: 0,
    totalContentLength: 0,
  }

  // Check minimum passage count
  if (passages.length < CONFIDENCE_CONFIG.minPassages) {
    return { isConfident: false, reason: 'insufficient_passages', signals }
  }

  // Check total content length
  signals.totalContentLength = passages.reduce((sum, p) => sum + p.content.length, 0)
  if (signals.totalContentLength < CONFIDENCE_CONFIG.minContentLength) {
    return { isConfident: false, reason: 'insufficient_content', signals }
  }

  // Calculate score-based signals using the score field from searchFiles
  // Score indicates similarity - higher is more relevant
  const scores = passages.map((p) => p.score)
  signals.avgScore = scores.reduce((a, b) => a + b, 0) / scores.length
  signals.highQualityCount = passages.filter(
    (p) => p.score >= CONFIDENCE_CONFIG.minPassageScore
  ).length

  // If no passages meet the minimum score threshold, confidence is low
  if (signals.highQualityCount === 0) {
    return { isConfident: false, reason: 'no_high_quality_passages', signals }
  }

  // Check for duplication (same file/source repeated)
  // Store raw ratio for logging, then compute boolean threshold
  signals.duplicateRatio = calculateDuplicationRatio(passages)
  signals.highDuplication = signals.duplicateRatio > CONFIDENCE_CONFIG.maxDuplicateRatio

  // Check for content similarity (near-duplicate text)
  signals.highContentSimilarity = hasHighContentSimilarity(passages)

  // If average score is very low, retrieval likely failed
  if (signals.avgScore < CONFIDENCE_CONFIG.lowScoreThreshold) {
    return { isConfident: false, reason: 'low_average_score', signals }
  }

  // If few passages AND high duplication, confidence is reduced
  // This catches cases where query matches only one document repeatedly
  if (passages.length <= 2 && signals.highDuplication) {
    return { isConfident: false, reason: 'few_passages_high_duplication', signals }
  }

  return { isConfident: true, reason: 'ok', signals }
}

/**
 * Creates an answer generation prompt that instructs the LLM to answer
 * ONLY from the provided passages and indicate when information is missing.
 */
const createAnswerPrompt = (
  question: string,
  passages: string[]
): gen.LLMInput => ({
  responseFormat: 'json_object',
  temperature: 0.1, // Low temperature for factual answers
  systemPrompt: `You are a helpful assistant that answers questions using ONLY the information provided in the passages below.

CRITICAL RULES:
1. Answer ONLY based on information explicitly stated in the passages
2. Do NOT make up or infer information not present in the passages
3. If the passages do not contain enough information to answer the question, set hasEnoughInfo to false
4. Be concise and direct in your answers
5. If multiple passages contain relevant information, synthesize them into a coherent answer

You must respond with a JSON object matching this schema:
${AnswerOutputSchema.toTypescriptType({ treatDefaultAsOptional: true })}

Where:
- "answer": Your answer based on the passages. If hasEnoughInfo is false, explain what information is missing.
- "hasEnoughInfo": true if you can confidently answer from the passages, false otherwise
- "confidence": "high" if answer is directly stated, "medium" if inferred from context, "low" if uncertain`,
  messages: [
    {
      role: 'user',
      content: `PASSAGES:
${passages.map((p, i) => `[${i + 1}] ${p}`).join('\n\n')}

QUESTION: ${question}

Please answer the question using ONLY the information in the passages above.`,
    },
  ],
})

const plugin = new bp.Plugin({
  actions: {},
})

plugin.on.beforeIncomingMessage('*', async ({ data: message, client, ctx, actions }) => {
  if (message.type !== 'text') {
    console.debug('Ignoring non-text message')
    return
  }

  const text: string = message.payload.text
  if (!text) {
    console.debug('Ignoring empty message')
    return
  }

  console.debug('Processing message:', text)

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 1: FETCH CONVERSATION CONTEXT (for multi-turn understanding)
  // Retrieve recent messages from the conversation to provide context for
  // follow-up questions like "what about pricing?" or "and refunds?".
  // This uses the existing client.listMessages() API - no external storage.
  // ─────────────────────────────────────────────────────────────────────────────
  let conversationContext = ''
  let recentMessages: Array<{ type: string; payload: unknown; userId: string }> = []
  
  try {
    // Fetch recent messages from this conversation for context
    const { messages: historyMessages } = await client.listMessages({
      conversationId: message.conversationId,
      tags: {},
    })
    
    // Take the most recent messages (excluding the current one being processed)
    // Filter to only include messages before the current one
    recentMessages = historyMessages
      .filter((m) => m.id !== message.id)
      .slice(0, CONTEXT_CONFIG.maxContextMessages)
      .reverse() // Oldest first for natural reading order
    
    conversationContext = formatConversationContext(recentMessages, ctx.botId)
    console.debug('Fetched conversation context:', recentMessages.length, 'messages')
  } catch (err) {
    // If we can't fetch history, proceed without context
    console.debug('Could not fetch conversation history:', err)
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 2: MULTI-TURN QUESTION RECONSTRUCTION
  // If the message appears to be referential/short (e.g., "what about that?"),
  // use LLM to reconstruct a standalone question using conversation context.
  // This happens BEFORE the regular question extraction for better accuracy.
  // ─────────────────────────────────────────────────────────────────────────────
  let standaloneQuestion = text
  let searchQuery = text
  let skipRegularExtraction = false

  // Check if message is likely referential and we have context to help
  if (isLikelyReferential(text) && conversationContext) {
    console.debug('Message appears referential, reconstructing standalone question...')
    
    try {
      const reconstructPrompt = createStandaloneQuestionPrompt(text, conversationContext)
      const reconstructOutput = await actions.llm.generateContent(reconstructPrompt)
      
      const { success: reconstructSuccess, json: reconstructJson } = gen.parseLLMOutput(reconstructOutput)
      if (reconstructSuccess) {
        const reconstructResult = StandaloneQuestionSchema.safeParse(reconstructJson)
        if (reconstructResult.success) {
          const { standaloneQuestion: reconstructed, searchQuery: query, isValidQuestion, needsContext } = reconstructResult.data
          
          if (!isValidQuestion) {
            console.debug('Not a valid question after reconstruction, skipping')
            return
          }
          
          // Use the reconstructed question for both search and answer generation
          standaloneQuestion = reconstructed
          searchQuery = query || reconstructed
          
          // If we successfully reconstructed, we can skip the regular extraction
          // since we already have a clean standalone question
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
  // If we didn't reconstruct, or if the message wasn't referential,
  // use the existing question extraction prompt with conversation context.
  // The prompt already handles context resolution via resolved_question.
  // ─────────────────────────────────────────────────────────────────────────────
  if (!skipRegularExtraction) {
    // Build the input for question extraction, including conversation context
    // The question prompt already supports context via formatQuestionExtractorMessage
    const llmInput = questions.prompt({ text, line: 'L1' })
    
    // Enhance the prompt with conversation context if available
    if (conversationContext && llmInput.messages.length > 0) {
      // Prepend context to the user message
      const lastMessage = llmInput.messages[llmInput.messages.length - 1]
      if (lastMessage && lastMessage.role === 'user') {
        lastMessage.content = `${conversationContext}\n\n${lastMessage.content}`
      }
    }
    
    const llmOutput = await actions.llm.generateContent(llmInput)

    const { success, json } = gen.parseLLMOutput(llmOutput)
    if (!success) {
      console.debug('Failed to extract questions')
      return
    }

    const parsedResult = questions.OutputFormat.safeParse(json)
    if (!parsedResult.success) {
      console.debug('Failed to extract questions')
      return
    }

    const { data } = parsedResult
    if (!data.hasQuestions || !data.questions?.length) {
      console.debug('No questions extracted')
      return
    }

    // Use the extracted search_query and resolved_question
    searchQuery = data.questions
      .map((q) => q.search_query || q.resolved_question)
      .join(' ')
    
    standaloneQuestion = data.questions
      .map((q) => q.resolved_question)
      .join(' ')

    console.debug('Extracted search query:', searchQuery)
    console.debug('Resolved question:', standaloneQuestion)
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 3.5: QUERY QUALITY CHECK
  // Before retrieval, check if the search query is valid.
  // Empty, very short, or stop-word-only queries produce poor retrieval.
  // Return the fallback early in these cases to avoid wasting resources.
  // ─────────────────────────────────────────────────────────────────────────────
  const queryQuality = evaluateQueryQuality(searchQuery)
  if (queryQuality.isWeak) {
    console.debug('Query quality too weak for reliable retrieval:', queryQuality.reason)
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: {
        text: CONFIDENCE_CONFIG.lowConfidenceFallback,
      },
      tags: {},
      type: 'text',
    })
    return { stop: true }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 4: RETRIEVAL
  // Use the standalone question/search query for retrieval.
  // The query is now context-aware for multi-turn conversations.
  // ─────────────────────────────────────────────────────────────────────────────
  console.debug('Searching for:', searchQuery)
  const { passages } = await client.searchFiles({
    query: searchQuery,
  })

  if (!passages.length) {
    console.debug('No passages found')
    return
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 5: CONFIDENCE HANDLING (Retrieval Quality)
  // Check if we have enough high-quality passages to answer confidently.
  // This evaluates:
  //   - Passage count and content length (basic requirements)
  //   - Relevance scores from searchFiles (quality signal)
  //   - Duplication ratio (same source repeated = weak retrieval)
  //   - Content similarity (near-duplicate text = weak diversity)
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

  // If retrieval confidence is too low, return fallback immediately
  if (!retrievalConfidence.isConfident) {
    console.debug('Low confidence in retrieval:', retrievalConfidence.reason)
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: {
        text: CONFIDENCE_CONFIG.lowConfidenceFallback,
      },
      tags: {},
      type: 'text',
    })
    return { stop: true }
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // STEP 6: ANSWER GENERATION
  // Generate an answer using the standalone question (context-aware).
  // The LLM is instructed to answer ONLY from the retrieved passages.
  // ─────────────────────────────────────────────────────────────────────────────
  const passageContents = passages.map((p) => p.content)
  
  try {
    // Use the standalone question for answer generation (includes context)
    const answerPrompt = createAnswerPrompt(standaloneQuestion, passageContents)
    const answerOutput = await actions.llm.generateContent(answerPrompt)
    
    const { success: answerSuccess, json: answerJson } = gen.parseLLMOutput(answerOutput)
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
    // Combine retrieval confidence signals with answer confidence to make the
    // final decision. Be stricter when:
    //   - Passages are few AND highly duplicative (limited source diversity)
    //   - Model indicates uncertainty (confidence = 'low' or 'medium' with weak signals)
    //   - hasEnoughInfo is false (model couldn't find answer in passages)
    //
    // The goal is to avoid answering when we're not confident, rather than
    // hallucinating or giving poor answers.
    // ─────────────────────────────────────────────────────────────────────────────
    const { signals } = retrievalConfidence

    // Case 1: Model explicitly says it doesn't have enough info
    if (!hasEnoughInfo) {
      console.debug('Model indicates insufficient information in passages')
      await client.createMessage({
        conversationId: message.conversationId,
        userId: ctx.botId,
        payload: {
          text: CONFIDENCE_CONFIG.lowConfidenceFallback,
        },
        tags: {},
        type: 'text',
      })
      return { stop: true }
    }

    // Case 2: Low answer confidence from the model
    if (answerConfidence === 'low') {
      console.debug('Answer confidence is low')
      await client.createMessage({
        conversationId: message.conversationId,
        userId: ctx.botId,
        payload: {
          text: CONFIDENCE_CONFIG.lowConfidenceFallback,
        },
        tags: {},
        type: 'text',
      })
      return { stop: true }
    }

    // Case 3: Medium confidence with weak retrieval signals = reject
    // This is the stricter gating: if the model is only medium confident
    // AND the passages were duplicative or had near-duplicate content,
    // we should not trust the answer.
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
          payload: {
            text: CONFIDENCE_CONFIG.lowConfidenceFallback,
          },
          tags: {},
          type: 'text',
        })
        return { stop: true }
      }
    }

    // Send the generated answer
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: {
        text: answer,
      },
      tags: {},
      type: 'text',
    })
  } catch (err) {
    console.error('Answer generation failed:', err)
    // Fallback: if answer generation fails, return the low confidence fallback
    await client.createMessage({
      conversationId: message.conversationId,
      userId: ctx.botId,
      payload: {
        text: CONFIDENCE_CONFIG.lowConfidenceFallback,
      },
      tags: {},
      type: 'text',
    })
  }

  return { stop: true }
})

export default plugin
