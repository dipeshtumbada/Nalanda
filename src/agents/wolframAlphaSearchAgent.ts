import { BaseMessage } from '@langchain/core/messages';
import {
  PromptTemplate,
  ChatPromptTemplate,
  MessagesPlaceholder,
} from '@langchain/core/prompts';
import {
  RunnableSequence,
  RunnableMap,
  RunnableLambda,
} from '@langchain/core/runnables';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { Document } from '@langchain/core/documents';
import { searchSearxng } from '../lib/searxng';
import type { StreamEvent } from '@langchain/core/tracers/log_stream';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { Embeddings } from '@langchain/core/embeddings';
import formatChatHistoryAsString from '../utils/formatHistory';
import eventEmitter from 'events';
import logger from '../utils/logger';

const basicWolframAlphaSearchRetrieverPrompt = `
You are Nalanda, an AI model built by Konect U specializing in creating and enhancing resumes for university applications. You will guide the user in gathering the necessary information to create a strong resume tailored to a specific university and program.

1. Ask the user to upload or paste their resume.
2. If the resume is not provided, prompt the user to share details one by one:
   a. Personal information (name, contact details)
   b. Target university and program
   c. Education history
   d. Work experience
   e. Skills
   f. Extracurricular activities
   g. Achievements and awards
   h. Volunteer work or community service

If a question is not relevant to this purpose, respond with "Could you please clarify your question to better assist with your resume?"

Follow up question: {query}
Rephrased question:
`;

const basicWolframAlphaSearchResponsePrompt = `
    You are Nalanda, an AI model built by Konect U specializing in creating and enhancing resumes for university applications. You are set on focus mode 'Resume Builder', meaning you will help the user create a strong resume tailored to a specific university and program.

Based on the provided information, proceed with the following steps:

1. If the user uploads a resume, analyze it for:
   a. Personal information and contact details
   b. Education history
   c. Work experience
   d. Skills
   e. Extracurricular activities
   f. Achievements and awards
   g. Volunteer work or community service
2. If the resume is not provided, ask for each component step by step.
3. Research the requirements and preferences of the specified university and course:
   a. Minimum GPA requirements
   b. Required or preferred coursework
   c. Desired skills or experiences
   d. Extracurricular expectations
   e. Unique attributes the program values
4. Compare the resume content to the university and course requirements.
5. Identify areas of strength:
   a. Experiences or achievements aligning with course requirements
   b. Relevant skills
   c. Academic performance meeting university standards
   d. Extracurricular activities showing leadership or commitment
6. Identify areas for improvement:
   a. Missing or weak elements for the chosen course
   b. Skills or experiences to enhance
   c. Gaps in the resume concerning admissions officers
7. Prepare feedback for the user:
   a. Summarize strong points and their value for the chosen university and course
   b. Suggest improvements with specific recommendations
   c. Additional elements to consider adding based on requirements
8. Ask the user if they need detailed advice on any specific aspect.
9. Offer suggestions for gaining relevant experiences or skills, if applicable.
10. Inquire if the user has questions about the feedback or needs clarification.
11. Recommend researching specific application requirements for the chosen university and course.
12. Suggest contacting the university's admissions office or attending information sessions for guidance.
13. Offer to review an updated resume if changes are made based on feedback.
14. Conclude by encouraging the user in their application process and reminding them that continuous improvement and tailoring of their resume can significantly enhance their chances of admission.

If a question is not relevant to this purpose, respond with "Could you please clarify your question to better assist with your resume?". Today's date is ${new Date().toISOString()}
`;

const strParser = new StringOutputParser();

const handleStream = async (
  stream: AsyncGenerator<StreamEvent, any, unknown>,
  emitter: eventEmitter,
) => {
  for await (const event of stream) {
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalSourceRetriever'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'sources', data: event.data.output }),
      );
    }
    if (
      event.event === 'on_chain_stream' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit(
        'data',
        JSON.stringify({ type: 'response', data: event.data.chunk }),
      );
    }
    if (
      event.event === 'on_chain_end' &&
      event.name === 'FinalResponseGenerator'
    ) {
      emitter.emit('end');
    }
  }
};

type BasicChainInput = {
  chat_history: BaseMessage[];
  query: string;
};

const createBasicWolframAlphaSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicWolframAlphaSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'en',
        engines: ['wolframalpha'],
      });

      const documents = res.results.map(
        (result) =>
          new Document({
            pageContent: result.content,
            metadata: {
              title: result.title,
              url: result.url,
              ...(result.img_src && { img_src: result.img_src }),
            },
          }),
      );

      return { query: input, docs: documents };
    }),
  ]);
};

const createBasicWolframAlphaSearchAnsweringChain = (llm: BaseChatModel) => {
  const basicWolframAlphaSearchRetrieverChain =
    createBasicWolframAlphaSearchRetrieverChain(llm);

  const processDocs = (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  return RunnableSequence.from([
    RunnableMap.from({
      query: (input: BasicChainInput) => input.query,
      chat_history: (input: BasicChainInput) => input.chat_history,
      context: RunnableSequence.from([
        (input) => ({
          query: input.query,
          chat_history: formatChatHistoryAsString(input.chat_history),
        }),
        basicWolframAlphaSearchRetrieverChain
          .pipe(({ query, docs }) => {
            return docs;
          })
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicWolframAlphaSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicWolframAlphaSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
) => {
  const emitter = new eventEmitter();

  try {
    const basicWolframAlphaSearchAnsweringChain =
      createBasicWolframAlphaSearchAnsweringChain(llm);
    const stream = basicWolframAlphaSearchAnsweringChain.streamEvents(
      {
        chat_history: history,
        query: query,
      },
      {
        version: 'v1',
      },
    );

    handleStream(stream, emitter);
  } catch (err) {
    emitter.emit(
      'error',
      JSON.stringify({ data: 'An error has occurred please try again later' }),
    );
    logger.error(`Error in WolframAlphaSearch: ${err}`);
  }

  return emitter;
};

const handleWolframAlphaSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = basicWolframAlphaSearch(message, history, llm);
  return emitter;
};

export default handleWolframAlphaSearch;
