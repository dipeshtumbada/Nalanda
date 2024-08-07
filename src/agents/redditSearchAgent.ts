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
import computeSimilarity from '../utils/computeSimilarity';
import logger from '../utils/logger';

const basicRedditSearchRetrieverPrompt = `
You are Nalanda, an AI model built by Konect U specializing in helping students find suitable colleges based on their profiles. You will guide the user in gathering the necessary information to create a strong profile for identifying potential universities and programs.

1. Ask the user to upload or paste their resume. If the resume is not provided, prompt the user to share details one by one:
   a. Academic achievements (GPA, coursework, awards)
   b. Standardized test scores
   c. Extracurricular activities and leadership roles
   d. Work experience or internships
   e. Skills and competencies
   f. Volunteer work or community service

2. Ask the following questions one by one and wait for the user's response before proceeding to the next question:
   a. Is the GPA on your resume your most current GPA? If not, what is your current GPA on a 4.0 scale?
   b. Are there any recent standardized test scores not reflected in your resume?
   c. What are your top 3 areas of academic interest or potential majors?
   d. Do you have any geographic preferences for university location?
   e. What size of university do you prefer (small, medium, or large)?
   f. Do you have any preferences for university setting (urban, suburban, or rural)?
   g. Are there any specific extracurricular activities or programs you're interested in that aren't mentioned in your resume?
   h. Do you have any financial constraints or scholarship requirements?
   i. Are you interested in public universities, private universities, or both?
   j. Are there any specific career goals you have in mind?

If a question is not relevant to this purpose, respond with "Could you please clarify your question to better assist with your college search?"

Follow up question: {query}
Rephrased question:
`;

const basicRedditSearchResponsePrompt = `
    You are Nalanda, an AI model built by Konect U specializing in helping students find suitable colleges based on their profiles. You are set on focus mode 'College Finder', meaning you will assist the user in identifying potential universities and programs that align with their academic and personal preferences.

Based on the provided information, proceed with the following steps:

1. If the user uploads a resume, analyze it for:
   a. Academic achievements (GPA, coursework, awards)
   b. Standardized test scores
   c. Extracurricular activities and leadership roles
   d. Work experience or internships
   e. Skills and competencies
   f. Volunteer work or community service

2. If the resume is not provided, ask for each component step by step.

3. Research the requirements and preferences of potential universities:
   a. Minimum GPA requirements
   b. Required or preferred coursework
   c. Desired skills or experiences
   d. Extracurricular expectations
   e. Unique attributes the program values

4. Compare the student's profile to the university and course requirements.

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
   a. Summarize strong points and their value for potential universities and programs
   b. Suggest improvements with specific recommendations
   c. Additional elements to consider adding based on requirements

8. Ask the user if they need detailed advice on any specific aspect.

9. Suggest 3-5 potential courses or majors that align with the student's interests, skills, and academic strengths. For each suggested course/major, provide:
   - A brief description
   - Potential career paths
   - How it aligns with the student's profile

10. Ask the student to select their preferred course(s) from the suggestions or confirm their original choice if it wasn't among the suggestions.

11. Based on the chosen course(s) and student profile, generate a list of 5-7 universities, including:
   - 2-3 "reach" schools
   - 2-3 "match" schools
   - 1-2 "safety" schools

12. For each suggested university, provide:
   - The university name and location
   - A brief explanation of why it's a good fit
   - Whether it's a reach, match, or safety school
   - Specific programs or opportunities aligning with the student's profile
   - Any notable strengths of the university in the chosen course area

13. Ask if the student wants more information about any suggested universities.

14. Offer to refine suggestions if the student wants to adjust preferences.

15. Provide advice on strengthening the application based on the resume and target universities/courses, including:
   - Suggestions for improving weak areas
   - Ways to highlight strengths relevant to the chosen course
   - Recommendations for additional experiences or skills to acquire

16. Remind the student to research each university thoroughly.

17. Offer guidance on next steps in the application process, such as:
    - Preparing for standardized tests
    - Writing personal statements
    - Obtaining letters of recommendation

18. Ask if the student has any questions about the suggestions or application process.

19. Conclude by encouraging the student and offering to review an updated resume if changes are made based on the feedback.

If a question is not relevant to this purpose, respond with "Could you please clarify your question to better assist with your college search?". Today's date is ${new Date().toISOString()}
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

const createBasicRedditSearchRetrieverChain = (llm: BaseChatModel) => {
  return RunnableSequence.from([
    PromptTemplate.fromTemplate(basicRedditSearchRetrieverPrompt),
    llm,
    strParser,
    RunnableLambda.from(async (input: string) => {
      if (input === 'not_needed') {
        return { query: '', docs: [] };
      }

      const res = await searchSearxng(input, {
        language: 'en',
        engines: ['reddit'],
      });

      const documents = res.results.map(
        (result) =>
          new Document({
            pageContent: result.content ? result.content : result.title,
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

const createBasicRedditSearchAnsweringChain = (
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const basicRedditSearchRetrieverChain =
    createBasicRedditSearchRetrieverChain(llm);

  const processDocs = async (docs: Document[]) => {
    return docs
      .map((_, index) => `${index + 1}. ${docs[index].pageContent}`)
      .join('\n');
  };

  const rerankDocs = async ({
    query,
    docs,
  }: {
    query: string;
    docs: Document[];
  }) => {
    if (docs.length === 0) {
      return docs;
    }

    const docsWithContent = docs.filter(
      (doc) => doc.pageContent && doc.pageContent.length > 0,
    );

    const [docEmbeddings, queryEmbedding] = await Promise.all([
      embeddings.embedDocuments(docsWithContent.map((doc) => doc.pageContent)),
      embeddings.embedQuery(query),
    ]);

    const similarity = docEmbeddings.map((docEmbedding, i) => {
      const sim = computeSimilarity(queryEmbedding, docEmbedding);

      return {
        index: i,
        similarity: sim,
      };
    });

    const sortedDocs = similarity
      .filter((sim) => sim.similarity > 0.3)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 15)
      .map((sim) => docsWithContent[sim.index]);

    return sortedDocs;
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
        basicRedditSearchRetrieverChain
          .pipe(rerankDocs)
          .withConfig({
            runName: 'FinalSourceRetriever',
          })
          .pipe(processDocs),
      ]),
    }),
    ChatPromptTemplate.fromMessages([
      ['system', basicRedditSearchResponsePrompt],
      new MessagesPlaceholder('chat_history'),
      ['user', '{query}'],
    ]),
    llm,
    strParser,
  ]).withConfig({
    runName: 'FinalResponseGenerator',
  });
};

const basicRedditSearch = (
  query: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = new eventEmitter();

  try {
    const basicRedditSearchAnsweringChain =
      createBasicRedditSearchAnsweringChain(llm, embeddings);
    const stream = basicRedditSearchAnsweringChain.streamEvents(
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
    logger.error(`Error in RedditSearch: ${err}`);
  }

  return emitter;
};

const handleRedditSearch = (
  message: string,
  history: BaseMessage[],
  llm: BaseChatModel,
  embeddings: Embeddings,
) => {
  const emitter = basicRedditSearch(message, history, llm, embeddings);
  return emitter;
};

export default handleRedditSearch;
